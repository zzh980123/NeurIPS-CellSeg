import torch.nn as nn

from models.swinunetrv2 import SwinUNETRV2

class SegGAN(nn.Module):

    def __init__(self, input_size, in_channels=3, classes_num=3):
        super().__init__()

        self.segment_net = SwinUNETRV2(input_size, in_channels=in_channels, out_channels=classes_num)
        self.dice_net = resnet50(in_channels=in_channels + classes_num, out_channels=1, img_size= input_size)