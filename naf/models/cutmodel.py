import monai.networks.nets as nets
import torch.nn as nn
class CutNet(nn.Module):
    def __init__(self, img_size, in_channels, classes_num):
        super().__init__()
        self.seg_net = nets.UNet(2, in_channels=in_channels, out_channels=1)
        self.cut_net = nets.ViT(in_channels=in_channels + 1, img_size=img_size, patch_size=2, num_classes=1, spatial_dims=2, num_layers=6)

    def forward(self, x):
        pass
