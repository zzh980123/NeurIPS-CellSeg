import torch.nn as nn
from torchvision import models

class SizeModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = models.vgg13_bn(pretrained=True)
        self.size_reg = nn.Conv2d()

