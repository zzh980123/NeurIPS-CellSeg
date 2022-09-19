from models.daformer import *
from models.coat import *
from models.daformer_ext import daformer_conv3x3x3


def criterion_aux_loss(logit, mask):
    mask = F.interpolate(mask, size=logit.shape[-2:], mode='nearest')
    loss = F.binary_cross_entropy_with_logits(logit, mask)
    return loss


class RGB(nn.Module):
    IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]  # [0.5, 0.5, 0.5]
    IMAGE_RGB_STD = [0.229, 0.224, 0.225]  # [0.5, 0.5, 0.5]

    def __init__(self, ):
        super(RGB, self).__init__()

        self.register_buffer('mean', torch.tensor(self.IMAGE_RGB_MEAN).view((1, 3, 1, 1)))
        self.register_buffer('std', torch.tensor(self.IMAGE_RGB_STD).view((1, 3, 1, 1)))

    def forward(self, x):
        return (x - self.mean) / self.std


class SizeRegressionDecoder(nn.Module):
    def __init__(self, in_channels, out_dims=9):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, padding=0)
        self.normal = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Linear(in_channels, out_dims)

    def forward(self, feature):
        x = self.conv(feature)
        x = self.relu(self.normal(x))
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.mlp(x)

        return x


class DaFormaerCoATNet_db(nn.Module):

    def __init__(self,
                 in_channel=3,
                 out_channel=3,
                 encoder=coat_lite_medium,
                 encoder_pretrain='coat_lite_medium_384x384_f9129688.pth',
                 decoder=daformer_conv3x3,
                 decoder_dim=320):
        super(DaFormaerCoATNet_db, self).__init__()

        self.rgb = RGB()

        self.encoder = encoder(in_channels=in_channel)

        encoder_dim = self.encoder.embed_dims

        self.decoder = decoder(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
        )
        self.logit = nn.Sequential(
            nn.Conv2d(decoder_dim, out_channel, kernel_size=1),
        )

        self.upsample = MixUpSample(scale_factor=4)

        # self.fine = nn.Sequential(
        #     nn.Conv2d(out_channel + in_channel, out_channel, kernel_size=1),
        #     nn.BatchNorm2d(out_channel),
        #     nn.LeakyReLU(inplace=True)
        # )

        # self.upsample2 = MixUpSample(scale_factor=2)
        # self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.sr = SizeRegressionDecoder(encoder_dim[-1], 9)

        # try to load the pretrained model of CoAT
        if encoder_pretrain is not None:
            checkpoint = torch.load(encoder_pretrain, map_location=lambda storage, loc: storage)
            self.encoder.load_state_dict(checkpoint['model'], strict=False)

        # segment : 1  classify size: 2
        self.mode = 2

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, x):
        if x.shape[1] == 3:
            x = self.rgb(x)
        encode_info = self.encoder(x)

        if self.mode == 1:
            last, decode_info = self.decoder(encode_info)

            # high_freq_info = self.downsample(x)
            logit = self.logit(last)

            upsample_logit = self.upsample(logit)

            # upsample_logit = self.fine(torch.cat([high_freq_info, upsample_logit], dim=1))
            #
            # upsample_logit = self.upsample2(upsample_logit)

            return upsample_logit
        elif self.mode == 2:
            feature = encode_info[-1]
            size_type = self.sr(feature)
            return size_type
        else:
            pass
