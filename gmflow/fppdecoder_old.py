# 整体代码为照搬，大体修改 ，不知道需不需要把BN换成IN 2025年9月6日 → #~ 已修改 2025年9月8日

import torch
import torch.nn as nn
import numpy as np

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        #↓ 修改成IN
        norm = nn.InstanceNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, norm, relu)

#由于下采样的通道数的问题， c=96的通道信息，即为从cnn出来的1/8信息和该信息再被transformer处理后的信息也是1/8信息
#c=96 信息得先拼接后采样
class DecoderBlock_catfirst(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        #!先拼接
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        # 再卷积
        x = self.conv1(x)
        x = self.conv2(x)
        #再上采样
        x = self.up(x)
        return x


# 2025年9月6日
class DecoderCup(nn.Module):
    #? hidden_size 
    # 从 Vision Transformer (ViT) 编码器输出的特征图的通道维度  
    # ↑ 作用: 它被用作解码器第一个二维卷积层 (conv_more) 的输入通道数 (in_channels)。
    # ↑ 该层的作用是将来自 ViT 的扁平化 patch embedding 转换为解码器后续部分可以处理的空间特征图。
    # ↑ #~ 对于 ViT-B 模型，hidden_size 的值通常是 768。我输出的是多少 ->128
    #? decoder_channels
    #↑  它定义了每个 DecoderBlock 的输出通道数。当解码器对图像进行上采样、提高空间分辨率时，通常会减少通道数。
    #↑  #~ 之前找到的 get_r50_b16_config() 函数将其设置为 (256, 128, 64, 16)。 我该如何设置 -> (128,96,64,16)
    #? n_skip
    # 实际使用多少个来自 CNN 主干网络的跳跃连接 它决定了哪些来自 ResNet 主干网络的特征图会被传递给解码器模块。
    # 你的 train.py 脚本将其设置为 3，意味着将使用三个层级的跳跃连接。
    # 代码中甚至有逻辑来处理不使用的跳跃连接，将其通道信息清零
    #? skip_channels
    # 定义了来自跳跃连接的每个特征图的通道数 它提供了将在每个 DecoderBlock 内部进行拼接的 ResNet 特征的通道维度。
    # 这个列表与 config.n_skip 参数协同工作
    #~ get_r50_b16_config() 函数将其设置为 [512, 256, 64, 16]。我该如何设置 -> [128,96,64,16] 最后一位在代码中设置成了0

    def __init__(self, hidden_size=128, decoder_channels=(128,96,64,32), n_skip=3,skip_channels=[128,96,64,32] ):
        super().__init__()

        self.n_skip = n_skip
        #↓ 将 head_channels 的大小与解码流程的第一个阶段对齐 512-—> #~128
        head_channels = 128

        if n_skip != 0:
            skip_channels = skip_channels
            for i in range(4-n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0
        else:
            skip_channels=[0,0,0,0]
        
        self.block1 = DecoderBlock_catfirst(
            in_channels=hidden_size,
            out_channels=decoder_channels[0],
            skip_channels=skip_channels[0]
        )

        in_channels_rem = list(decoder_channels[:-1])
        out_channels_rem = list(decoder_channels[1:])
        skip_channels_rem = list(skip_channels[1:])

        self.blocks_remaining = nn.ModuleList([
            DecoderBlock_catfirst(inch,outch,skch) for inch,outch,skch in zip(in_channels_rem, out_channels_rem, skip_channels_rem)
            ])


    def forward(self, hidden_states, features=None):
        #B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        #h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        # x = hidden_states.permute(0, 2, 1)
        # x = x.contiguous().view(B, hidden, h, w)
        # ↑ 上两行没必要在这里调整形状
        x = hidden_states # 在这里不用调整形状

        #使用第一个跳跃连接块
        x = self.block1(x,features[0])
 
        # 循环处理剩下的跳跃连接
        for i, block in enumerate(self.blocks_remaining):
            # features[1] 用于第二个块, features[2] 用于第三个块...
            skip = features[i+1] if (i+1 < self.n_skip) else None
            x = block(x, skip)
        return x


# ↓ 为了测试形状单独使用 2025年9月8日
if __name__ == "__main__":
    from print_model_info import print_conv_kernel_info
    from torchinfo import summary
    batchsize = 12
    model = DecoderCup()
    feature1_size = (batchsize, 128, 64, 64)
    features_size = [
        (batchsize, 128, 64, 64),
        (batchsize, 96, 128, 128),
        (batchsize, 64, 256, 256)
    ]

    hidden_states_tensor = torch.randn(*feature1_size)
    features_list_of_tensors = [torch.randn(*size) for size in features_size]

    # col_names 用于更清晰地显示
    summary(model, input_data={'hidden_states':hidden_states_tensor,'features':features_list_of_tensors}, col_names=["input_size", "output_size", "num_params", "mult_adds"])

    #print_conv_kernel_info(model)