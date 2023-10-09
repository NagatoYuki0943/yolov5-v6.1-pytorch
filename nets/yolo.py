import torch
import torch.nn as nn

from nets.CSPdarknet import C3, Conv, CSPDarknet


#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, pretrained=False):
        super(YoloBody, self).__init__()
        depth_dict          = {'n': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict          = {'n': 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]

        base_channels       = int(wid_mul * 64)  # 64
        base_depth          = max(round(dep_mul * 3), 1)  # 3
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        #-----------------------------------------------#
        self.backbone       = CSPDarknet(base_channels, base_depth, phi, pretrained)

        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        #-------------------------------------------#
        #   P5开始的的1x1Conv,得到 P5
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        self.conv_for_feat3         = Conv(base_channels * 16, base_channels * 8, 1, 1)
        #-------------------------------------------#
        #   P5_upsample,feat2 拼接之后的 CSPLayer
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        self.conv3_for_upsample1    = C3(base_channels * 16, base_channels * 8, base_depth, shortcut=False)

        #-------------------------------------------#
        #   拼接 P5_upsample,feat2 后的CSPLayer的1x1Conv,得到P4
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        self.conv_for_feat2         = Conv(base_channels * 8, base_channels * 4, 1, 1)
        #-------------------------------------------#
        #   拼接 P4_upsample,feat1 后的 CPSLayer,得到 P3_out
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        self.conv3_for_upsample2    = C3(base_channels * 8, base_channels * 4, base_depth, shortcut=False)

        #-------------------------------------------#
        #   P3_out下采样
        #   80, 80, 256 -> 40, 40, 256
        #-------------------------------------------#
        self.down_sample1           = Conv(base_channels * 4, base_channels * 4, 3, 2)
        #-------------------------------------------#
        #   拼接 P3_downsample, P4 后的 CPSLayer,得到 P4_out
        #   40, 40, 512 -> 40, 40, 512
        #-------------------------------------------#
        self.conv3_for_downsample1  = C3(base_channels * 8, base_channels * 8, base_depth, shortcut=False)

        #-------------------------------------------#
        #   P4_out下采样
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        self.down_sample2           = Conv(base_channels * 8, base_channels * 8, 3, 2)
        #-------------------------------------------#
        #   拼接 P4_downsample, P5 后的 CPSLayer,得到 P5_out
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        self.conv3_for_downsample2  = C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False)

        #-------------------------------------------#
        #   yolo_head
        #   3个1x1Conv
        #-------------------------------------------#
        # 80, 80, 256 => 80, 80, 3 * (5 + num_classes) => 80, 80, 3 * (4 + 1 + num_classes)
        self.yolo_head_P3 = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * (5 + num_classes), 1)
        # 40, 40, 512 => 40, 40, 3 * (5 + num_classes) => 40, 40, 3 * (4 + 1 + num_classes)
        self.yolo_head_P4 = nn.Conv2d(base_channels * 8, len(anchors_mask[1]) * (5 + num_classes), 1)
        # 20, 20, 1024 => 20, 20, 3 * (5 + num_classes) => 20, 20, 3 * (4 + 1 + num_classes)
        self.yolo_head_P5 = nn.Conv2d(base_channels * 16, len(anchors_mask[0]) * (5 + num_classes), 1)

    def forward(self, x):
        #-------------------------------------------#
        #   feat1: 80, 80, 256
        #   feat2: 40, 40, 512
        #   feat3: 20, 20, 1024
        #-------------------------------------------#
        feat1, feat2, feat3 = self.backbone(x)

        """第一次上采样"""
        #-------------------------------------------#
        #   P5开始的的1x1Conv,得到 P5
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        P5          = self.conv_for_feat3(feat3)
        #-------------------------------------------#
        #   P5上采样
        #   20, 20, 512 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.upsample(P5)
        #-------------------------------------------#
        #   拼接 P5_upsample,feat2
        #   40, 40, 512 cat 40, 40, 512 -> 40, 40, 1024
        #-------------------------------------------#
        P4          = torch.cat([P5_upsample, feat2], 1)
        #-------------------------------------------#
        #   P5_upsample,feat2 拼接之后的CSPLayer
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        P4          = self.conv3_for_upsample1(P4)

        """第二次上采样"""
        #-------------------------------------------#
        #   拼接 P5_upsample,feat2 后的CSPLayer的1x1Conv,得到P4
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        P4          = self.conv_for_feat2(P4)
        #-------------------------------------------#
        #   P4的上采样
        #   40, 40, 256 -> 80, 80, 256
        #-------------------------------------------#
        P4_upsample = self.upsample(P4)
        #-------------------------------------------#
        #   拼接 P4_upsample,feat1
        #   80, 80, 256 cat 80, 80, 256 -> 80, 80, 512
        #-------------------------------------------#
        P3          = torch.cat([P4_upsample, feat1], 1)
        #-------------------------------------------#
        #   拼接 P4_upsample,feat1 后的 CPSLayer,得到 P3_out
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        P3_out      = self.conv3_for_upsample2(P3)

        """第一次下采样"""
        #-------------------------------------------#
        #   P3_out下采样
        #   80, 80, 256 -> 40, 40, 256
        #-------------------------------------------#
        P3_downsample = self.down_sample1(P3_out)
        #-------------------------------------------#
        #   拼接 P3_downsample, P4
        #   40, 40, 256 cat 40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        P4            = torch.cat([P3_downsample, P4], 1)
        #-------------------------------------------#
        #   拼接 P3_downsample, P4 后的 CPSLayer,得到 P4_out
        #   40, 40, 512 -> 40, 40, 512
        #-------------------------------------------#
        P4_out        = self.conv3_for_downsample1(P4)

        """第二次下采样"""
        #-------------------------------------------#
        #   P4_out下采样
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        P4_downsample = self.down_sample2(P4_out)
        #-------------------------------------------#
        #   拼接 P4_downsample, P5
        #   20, 20, 512 cat 20, 20, 512 -> 20, 20, 1024
        #-------------------------------------------#
        P5            = torch.cat([P4_downsample, P5], 1)
        #-------------------------------------------#
        #   拼接 P4_downsample, P5 后的 CPSLayer,得到 P5_out
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        P5_out        = self.conv3_for_downsample2(P5)

        #-------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,75,80,80)
        #-------------------------------------------#
        out2 = self.yolo_head_P3(P3_out)
        #-------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,75,40,40)
        #-------------------------------------------#
        out1 = self.yolo_head_P4(P4_out)
        #-------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,75,20,20)
        #-------------------------------------------#
        out0 = self.yolo_head_P5(P5_out)
        return out0, out1, out2
