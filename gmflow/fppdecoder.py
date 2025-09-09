import torch
import torch.nn as nn
import numpy as np

class Conv2dReLU(nn.Sequential):
    """
    卷积+实例归一化+ReLU激活的组合模块
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        padding: 填充大小
        stride: 步长
        use_batchnorm: 是否使用归一化
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        # 创建卷积层
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,  # 如果使用归一化，则不需要偏置
        )
        
        # 创建实例归一化层
        norm = nn.InstanceNorm2d(out_channels)
        
        # 创建ReLU激活函数
        relu = nn.ReLU(inplace=True)
        
        # 按顺序组合各层
        super(Conv2dReLU, self).__init__(conv, norm, relu)


class DecoderBlockCatFirst(nn.Module):
    """
    解码器块，先拼接跳跃连接再进行卷积和上采样
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        skip_channels: 跳跃连接的通道数
        use_batchnorm: 是否使用归一化
        upsample: 是否进行上采样
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
            upsample=True,
    ):
        super().__init__()
        
        # 第一个卷积块
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,  # 输入通道数加上跳跃连接的通道数
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        
        # 第二个卷积块
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        
        # 上采样层
        self.upsample = upsample
        if upsample:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        # 如果有跳跃连接，先进行拼接
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        
        # 两次卷积操作
        x = self.conv1(x)
        x = self.conv2(x)
        
        # 如果需要上采样
        if self.upsample:
            x = self.up(x)
        
        return x


class DecoderCup(nn.Module):
    """
    解码器模块，处理来自Transformer的特征和CNN的跳跃连接
    
    参数:
        hidden_size: Transformer输出的特征维度
        decoder_channels: 解码器各层的通道数
        n_skip: 使用的跳跃连接数量
        skip_channels: 跳跃连接的通道数列表
    """
    def __init__(
        self, 
        hidden_size=128, 
        decoder_channels=(96, 64, 32,16), 
        n_skip=3,
        skip_channels=[128, 96, 64, 32]
    ):
        super().__init__()
        
        self.n_skip = n_skip
        
        # 处理跳跃连接通道数
        # 将不需要的跳跃连接通道数设置为0
        processed_skip_channels = skip_channels.copy()
        for i in range(4 - n_skip):
            processed_skip_channels[3 - i] = 0
        
        # 创建所有解码块，包括第一个和剩余的
        self.blocks = nn.ModuleList()
        
        # 第一个解码块
        self.blocks.append(
            DecoderBlockCatFirst(
                in_channels=hidden_size,
                out_channels=decoder_channels[0],
                skip_channels=processed_skip_channels[0],
                upsample=True  # 第一个块需要上采样
            )
        )
        
        # 中间的解码块
        for i in range(1, len(decoder_channels) - 1):
            self.blocks.append(
                DecoderBlockCatFirst(
                    in_channels=decoder_channels[i-1],
                    out_channels=decoder_channels[i],
                    skip_channels=processed_skip_channels[i],
                    upsample=True  # 中间块需要上采样
                )
            )
        
        # 最后一个解码块（不上采样）
        self.blocks.append(
            DecoderBlockCatFirst(
                in_channels=decoder_channels[-2],
                out_channels=decoder_channels[-1],
                skip_channels=processed_skip_channels[-1],
                upsample=False  # 最后一个块不上采样
            )
        )

    def forward(self, hidden_states, features=None):
        """
        前向传播
        
        参数:
            hidden_states: Transformer输出的特征
            features: 来自CNN的跳跃连接特征列表
            
        返回:
            解码后的特征
        """
        x = hidden_states
        
        # 处理所有解码块
        for i, block in enumerate(self.blocks):
            # 获取对应的跳跃连接（如果有）
            skip_feature = None
            if features and i < len(features):
                skip_feature = features[i]
            
            x = block(x, skip_feature)
        
        return x


# 测试代码
if __name__ == "__main__":
    from print_model_info import print_conv_kernel_info
    from torchinfo import summary
    
    batch_size = 12
    model = DecoderCup()
    
    # 定义输入形状
    feature_sizes = [
        (batch_size, 128, 64, 64),    # 来自Transformer的特征
        (batch_size, 128, 64, 64),    # 跳跃连接1
        (batch_size, 96, 128, 128),   # 跳跃连接2
        (batch_size, 64, 256, 256)    # 跳跃连接3
    ]
    
    # 创建随机输入
    inputs = [torch.randn(*size) for size in feature_sizes]
    hidden_states = inputs[0]
    features = inputs[1:]  # 跳过第一个，它是hidden_states
    
    # 使用torchinfo显示模型信息
    summary(
        model, 
        input_data={'hidden_states': hidden_states, 'features': features}, 
        col_names=["input_size", "output_size", "num_params", "mult_adds"]
    )
    
    # 测试模型前向传播
    with torch.no_grad():
        output = model(hidden_states, features)
        print(f"Output shape: {output.shape}")