import monai.networks.nets as nets
import torch.nn as nn
import torch


class CutNet(nn.Module):
    def __init__(self, img_size, in_channels, classes_num=1):
        super().__init__()
        self.seg_net = nets.UNet(2, in_channels=in_channels,
                                 out_channels=1)
        self.cut_net = nets.UNet(2, in_channels=in_channels,
                                 out_channels=1,
                                 channels=(4, 8, 16),
                                 strides=(2, 2),
                                 num_res_units=2)

    def forward(self, x):
        inout_map = self.seg_net(x)

        # contains contours
        inout_map = torch.sigmoid(inout_map)

        y = inout_map.detach() * x

        z = self.cut_net(y)

        # not contains contours
        z = torch.sigmoid(z)
        in_map = inout_map * (1 - z)

        return z, in_map
