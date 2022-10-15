import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.layers import DeformableConvLayer, HugeConv2dBlock


class MixUpSample(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.mixing = nn.Parameter(torch.tensor(0.5))
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.mixing * F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False) \
            + (1 - self.mixing) * F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        return x


# https://github.com/lhoyer/DAFormer/blob/master/mmseg/models/decode_heads/daformer_head.py
def Conv2dBnReLU(in_channel, out_channel, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
    )


class ASPP(nn.Module):

    def __init__(self,
                 in_channel,
                 channel,
                 dilation,
                 ):
        super(ASPP, self).__init__()

        self.conv = nn.ModuleList()
        for d in dilation:
            self.conv.append(
                Conv2dBnReLU(
                    in_channel,
                    channel,
                    kernel_size=1 if d == 1 else 3,
                    dilation=d,
                    padding=0 if d == 1 else d,
                )
            )

        self.out = Conv2dBnReLU(
            len(dilation) * channel,
            channel,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        aspp = []
        for conv in self.conv:
            aspp.append(conv(x))
        aspp = torch.cat(aspp, dim=1)
        out = self.out(aspp)
        return out


# DepthwiseSeparable
class DSConv2d(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1
                 ):
        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DSASPP(nn.Module):

    def __init__(self,
                 in_channel,
                 channel,
                 dilation,
                 ):
        super(DSASPP, self).__init__()

        self.conv = nn.ModuleList()
        for d in dilation:
            if d == 1:
                self.conv.append(
                    Conv2dBnReLU(
                        in_channel,
                        channel,
                        kernel_size=1 if d == 1 else 3,
                        dilation=d,
                        padding=0 if d == 1 else d,
                    )
                )
            else:
                self.conv.append(
                    DSConv2d(
                        in_channel,
                        channel,
                        kernel_size=3,
                        dilation=d,
                        padding=d,
                    )
                )

        self.out = Conv2dBnReLU(
            len(dilation) * channel,
            channel,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        aspp = []
        for conv in self.conv:
            aspp.append(conv(x))
        aspp = torch.cat(aspp, dim=1)
        out = self.out(aspp)
        return out


class DaformerDecoder(nn.Module):
    def __init__(
            self,
            encoder_dim=[32, 64, 160, 256],
            decoder_dim=256,
            dilation=[1, 6, 12, 18],
            use_bn_mlp=True,
            fuse='conv3x3',
            img_size=None
    ):
        super().__init__()
        self.mlp = nn.ModuleList([
            nn.Sequential(
                # Conv2dBnReLU(dim, decoder_dim, 1, padding=0), #follow mmseg to use conv-bn-relu
                *(
                    (nn.Conv2d(dim, decoder_dim, 1, padding=0, bias=False),
                     nn.BatchNorm2d(decoder_dim),
                     nn.ReLU(inplace=True),
                     ) if use_bn_mlp else
                    (nn.Conv2d(dim, decoder_dim, 1, padding=0, bias=True),)
                ),

                MixUpSample(2 ** i) if i != 0 else nn.Identity(),
            ) for i, dim in enumerate(encoder_dim)])

        if fuse == 'conv1x1':
            self.fuse = nn.Sequential(
                nn.Conv2d(len(encoder_dim) * decoder_dim, decoder_dim, 1, padding=0, bias=False),
                nn.BatchNorm2d(decoder_dim),
                nn.ReLU(inplace=True),
            )

        if fuse == 'conv3x3':
            self.fuse = nn.Sequential(
                nn.Conv2d(len(encoder_dim) * decoder_dim, decoder_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(decoder_dim),
                nn.ReLU(inplace=True),
            )

        if fuse == 'aspp':
            self.fuse = ASPP(
                decoder_dim * len(encoder_dim),
                decoder_dim,
                dilation,
            )

        if fuse == 'ds-aspp':
            self.fuse = DSASPP(
                decoder_dim * len(encoder_dim),
                decoder_dim,
                dilation,
            )

        if fuse == 'conv3x3x3':
            self.fuse = nn.Sequential(
                nn.Conv3d(decoder_dim, decoder_dim, (4, 3, 3), padding=(0, 1, 1), stride=(4, 1, 1), bias=False, padding_mode='reflect'),
                # nn.Conv3d(2, 1, (4, 1, 1), padding=(2, 0, 0), stride=(1, 1, 1), bias=False, padding_mode='reflect'),
                nn.BatchNorm3d(decoder_dim),
                nn.ReLU(inplace=True),
            )

        if fuse == 'huge-conv':
            self.fuse = nn.Sequential(
                HugeConv2dBlock(decoder_dim, decoder_dim, kernel_size=(31, 31)),
                nn.ReLU(inplace=True)
            )

        # if fuse == 'dfc_v5':
        #     self.fuse = DeformableConvLayer(
        #         img_size=img_size,
        #         in_channel=decoder_dim * len(encoder_dim),
        #         out_channel=decoder_dim,
        #         stride=1,
        #         in_channel_fold=False
        #     )

    def forward(self, feature):

        out = []
        for i, f in enumerate(feature):
            f = self.mlp[i](f)
            out.append(f)

        ll = len(out)

        combine2d = torch.cat(out, dim=1)
        B, C, H, W = combine2d.shape
        assert C % ll == 0

        # shuffle the channels
        combine2d = combine2d.permute(0, 1, 3, 2).reshape(B, 1, W * C, H).reshape(B, 1 * ll, W * C // ll, H).permute(0, 1, 3, 2) \
            .reshape(B, 1, H * ll, W * C // ll).permute(0, 1, 3, 2).reshape(B, 1 * C // ll, W, H * ll).permute(0, 1, 3, 2)\
            .reshape(B, 1, C, H, W).reshape(B, C // ll, ll, H, W)

        #  B, C // ll, ll, H, W -> B, C // ll, 1, H, W
        x = self.fuse(combine2d) \
            .reshape(B, C // ll, H, W)  # B, C // ll, H, W

        return x, out


class daformer_conv3x3x3(DaformerDecoder):
    def __init__(self, **kwargs):
        super(daformer_conv3x3x3, self).__init__(
            fuse='conv3x3x3',
            **kwargs
        )
