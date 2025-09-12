import torch.nn as nn

from .trident_conv import MultiScaleTridentConv
#↓ 为了单独测试模块输出与形状使用 别忘了 手动切换 2025年9月8日
#from trident_conv import MultiScaleTridentConv 
#↑ 为了单独测试模块输出与形状使用

class ResidualBlock(nn.Module):
    # in_planes: 输入特征图的通道数（planes 在这里是通道 channel 的意思）。
    # planes: 在这个块内部以及最终输出的特征图的通道数。
    # dilation=1: 卷积核的膨胀率，用于扩大感受野而不增加参数，常用于分割任务 dilation=1 就是普通卷积
    def __init__(self, in_planes, planes, norm_layer=nn.InstanceNorm2d, stride=1, dilation=1,
                 ):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               dilation=dilation, padding=dilation, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               dilation=dilation, padding=dilation, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = norm_layer(planes)
        self.norm2 = norm_layer(planes)
        if not stride == 1 or in_planes != planes:
            self.norm3 = norm_layer(planes)

        if stride == 1 and in_planes == planes: # 步长和输入输出大小来确定是否在下采样，下采样的方式就是标准的步长变化
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class CNNEncoder(nn.Module):
    def __init__(self, output_dim=128,
                 norm_layer=nn.InstanceNorm2d,
                 num_output_scales=1,
                 **kwargs,
                 ):
        super(CNNEncoder, self).__init__()
        self.num_branch = num_output_scales

        feature_dims = [64, 96, 128]

        self.conv1 = nn.Conv2d(3, feature_dims[0], kernel_size=7, stride=2, padding=3, bias=False)  # 1/2 #todo
        self.norm1 = norm_layer(feature_dims[0])
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = feature_dims[0]
        self.layer1 = self._make_layer(feature_dims[0], stride=1, norm_layer=norm_layer)  # 1/2
        self.layer2 = self._make_layer(feature_dims[1], stride=2, norm_layer=norm_layer)  # 1/4

        # highest resolution 1/4 or 1/8
        stride = 2 if num_output_scales == 1 else 1
        self.layer3 = self._make_layer(feature_dims[2], stride=stride,
                                       norm_layer=norm_layer,
                                       )  # 1/4 or 1/8

        self.conv2 = nn.Conv2d(feature_dims[2], output_dim, 1, 1, 0)

        if self.num_branch > 1:
            if self.num_branch == 4:
                strides = (1, 2, 4, 8)
            elif self.num_branch == 3:
                strides = (1, 2, 4)
            elif self.num_branch == 2:
                strides = (1, 2)
            else:
                raise ValueError

            self.trident_conv = MultiScaleTridentConv(output_dim, output_dim,
                                                      kernel_size=3,
                                                      strides=strides,
                                                      paddings=1,
                                                      num_branch=self.num_branch,
                                                      )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d):
        layer1 = ResidualBlock(self.in_planes, dim, norm_layer=norm_layer, stride=stride, dilation=dilation)
        layer2 = ResidualBlock(dim, dim, norm_layer=norm_layer, stride=1, dilation=dilation)

        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)
    
#修改增加 skip_feature ，输出为一个list，[0]是正常的前向输出 [1]是跳跃输出2025年9月5日
    def forward(self, x):
        skip_feature=[]
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)  # 1/2
        skip_feature.append(x)
        x = self.layer2(x)  # 1/4
        skip_feature.append(x)
        x = self.layer3(x)  # 1/8 or 1/4
        skip_feature.append(x)

        x = self.conv2(x)

        if self.num_branch > 1:
            out = self.trident_conv([x] * self.num_branch)  # high to low res
        else:
            out = [x]

        return [out,skip_feature]


# ↓ 为了测试形状单独使用 2025年9月8日
if __name__ == "__main__":
    from print_model_info import print_conv_kernel_info
    from torchinfo import summary
    # bactsize = 12
    model = CNNEncoder()
    summary(model, input_size=(3, 512, 512))
    #print_conv_kernel_info(model)