import warnings

import torch
import torch.nn as nn


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

#----------------------------------------#
#   Conv + bn + act
#----------------------------------------#
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

#--------------------------------------------------#
#   残差结构的构建，小的残差结构
#   在这里通道和宽高都不变
#   两层卷积 1x1和3x3
#--------------------------------------------------#
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        #--------------------------------------------------#
        #   利用1x1卷积进行通道数的缩减。缩减率一般是50%
        #--------------------------------------------------#
        self.cv1 = Conv(c1, c_, 1, 1)
        #--------------------------------------------------#
        #   利用3x3卷积进行通道数的拓张。并且完成特征提取
        #--------------------------------------------------#
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

#--------------------------------------------------------------------#
#   CSPdarknet的结构块,每个stage的结构
#
#   CSPnet结构并不算复杂，就是将原来的残差块的堆叠进行了一个拆分，拆成左右两部分:
#   主干部分继续进行原来的残差块的堆叠；
#   另一部分则像一个残差边一样，经过少量处理直接连接到最后。
#   因此可以认为CSP中存在一个大的残差边。
#
#   V4中先进行了一次卷积让通道翻倍,宽高减半,这里没有做,而是分出去做了,直接就是分为两个分支了,所以最终宽高不变,维度也不变
#--------------------------------------------------------------------#
class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels

        #----------------------------------------#
        #   右侧
        #----------------------------------------#
        self.cv1 = Conv(c1, c_, 1, 1)
        #----------------------------------------#
        #   左侧
        #----------------------------------------#
        self.cv2 = Conv(c1, c_, 1, 1)
        #----------------------------------------#
        #   最终拼接
        #----------------------------------------#
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        #--------------------------------------------------#
        #   根据循环的次数构建上述Bottleneck残差结构
        #--------------------------------------------------#
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        #----------------------------------------#
        #   拼接两侧的输出
        #----------------------------------------#
        return self.cv3(torch.cat(
            (
                self.m(self.cv1(x)),
                self.cv2(x)
            )
            , dim=1))

#---------------------------------------------------#
#   SPP结构，利用不同大小的池化核进行池化,增大感受野
#   池化后和输入数据进行维度堆叠
#   pool_sizes=[1, 5, 9, 13] 1不变,所以不用做了
#---------------------------------------------------#
class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        # 1x1Conv缩减通道
        self.cv1 = Conv(c1, c_, 1, 1)
        # 1x1Conv调整最终通道
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class CSPDarknet(nn.Module):
    def __init__(self, base_channels, base_depth, phi, pretrained):
        super().__init__()
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道base_channels是64
        #-----------------------------------------------#

        #-----------------------------------------------#
        #   利用卷积结构进行特征提取
        #   640, 640, 3 -> 320, 320, 64
        #-----------------------------------------------#
        self.stem       = Conv(3, base_channels, 6, 2, 2)

        #-----------------------------------------------#
        #   完成卷积之后，320, 320, 64 -> 160, 160, 128
        #   完成CSPlayer之后，160, 160, 128 -> 160, 160, 128
        #-----------------------------------------------#
        self.dark2 = nn.Sequential(
            # 320, 320, 64 -> 160, 160, 128
            Conv(base_channels, base_channels * 2, 3, 2),
            # 160, 160, 128 -> 160, 160, 128
            C3(base_channels * 2, base_channels * 2, base_depth),
        )

        #-----------------------------------------------#
        #   完成卷积之后，160, 160, 128 -> 80, 80, 256
        #   完成CSPlayer之后，80, 80, 256 -> 80, 80, 256
        #                   在这里引出有效特征层80, 80, 256
        #                   进行加强特征提取网络FPN的构建
        #-----------------------------------------------#
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C3(base_channels * 4, base_channels * 4, base_depth * 2),
        )

        #-----------------------------------------------#
        #   完成卷积之后，80, 80, 256 -> 40, 40, 512
        #   完成CSPlayer之后，40, 40, 512 -> 40, 40, 512
        #                   在这里引出有效特征层40, 40, 512
        #                   进行加强特征提取网络FPN的构建
        #-----------------------------------------------#
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C3(base_channels * 8, base_channels * 8, base_depth * 3),
        )

        #-----------------------------------------------#
        #   完成卷积之后，40, 40, 512 -> 20, 20, 1024
        #   完成SPP之后，20, 20, 1024 -> 20, 20, 1024
        #   完成CSPlayer之后，20, 20, 1024 -> 20, 20, 1024
        #-----------------------------------------------#
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2),
            C3(base_channels * 16, base_channels * 16, base_depth),
            SPPF(base_channels * 16, base_channels * 16),
        )
        if pretrained:
            backbone = "cspdarknet_" + phi
            url = {
                "cspdarknet_n" : 'https://github.com/bubbliiiing/yolov5-v6.1-pytorch/releases/download/v1.0/cspdarknet_n_v6.1_backbone.pth',
                "cspdarknet_s" : 'https://github.com/bubbliiiing/yolov5-v6.1-pytorch/releases/download/v1.0/cspdarknet_s_v6.1_backbone.pth',
                'cspdarknet_m' : 'https://github.com/bubbliiiing/yolov5-v6.1-pytorch/releases/download/v1.0/cspdarknet_m_v6.1_backbone.pth',
                'cspdarknet_l' : 'https://github.com/bubbliiiing/yolov5-v6.1-pytorch/releases/download/v1.0/cspdarknet_l_v6.1_backbone.pth',
                'cspdarknet_x' : 'https://github.com/bubbliiiing/yolov5-v6.1-pytorch/releases/download/v1.0/cspdarknet_x_v6.1_backbone.pth',
            }[backbone]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from ", url.split('/')[-1])

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        #-----------------------------------------------#
        #   dark3的输出为80, 80, 256，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        #-----------------------------------------------#
        #   dark4的输出为40, 40, 512，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark4(x)
        feat2 = x
        #-----------------------------------------------#
        #   dark5的输出为20, 20, 1024，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3
