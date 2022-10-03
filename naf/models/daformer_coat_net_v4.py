from models.daformer import *
from models.coat import *
from models.layers import PixelShuffleBlock


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


class DaFormaerCoATNet_GRAD_V4(nn.Module):

    def __init__(self,
                 in_channel=3,
                 out_channel=4,
                 classes_num=4,            # cell images has 4 modalities
                 encoder=coat_lite_medium,
                 encoder_pretrain='coat_lite_medium_384x384_f9129688.pth',
                 decoder=daformer_involution,
                 decoder_dim=320):
        super(DaFormaerCoATNet_GRAD_V4, self).__init__()

        assert out_channel > 3
        # channel0: in prob
        # channel1: out prob
        # channel2: grad y
        # channel3: grad x

        self.rgb = RGB()

        self.encoder = encoder(in_channels=in_channel)

        encoder_dim = self.encoder.embed_dims

        self.class_encoder = nn.Sequential(
            Conv2dBnReLU(encoder_dim, encoder_dim // 2),
            nn.AdaptiveAvgPool2d(1),
            # cell images has 4 modalities
            nn.Conv2d(in_channels=encoder_dim // 2, out_channels=classes_num, kernel_size=1),
            nn.BatchNorm2d(classes_num),
            nn.LeakyReLU(inplace=True)
        )

        # shuffle the channels
        self.class_linear = nn.Linear(
            decoder_dim,
            decoder_dim
        )

        self.decoder = decoder(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
        )

        self.logit = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_dim),
            PixelShuffleBlock(decoder_dim, out_channel - 2, upscale_factor=4),
            nn.BatchNorm2d(out_channel - 2),
        )

        self.grad_conv = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_dim),
            PixelShuffleBlock(decoder_dim, 2, upscale_factor=4),
            nn.BatchNorm2d(2)
        )

        # self.conv = nn.Sequential(
        #     Conv2dBnReLU(in_channel=in_channel, out_channel=decoder_dim, kernel_size=5, padding=2, stride=1),
        #     Conv2dBnReLU(in_channel=decoder_dim, out_channel=decoder_dim, kernel_size=5, padding=2, stride=1),
        #     Conv2dBnReLU(in_channel=decoder_dim, out_channel=out_channel, kernel_size=5, padding=2, stride=1),
        # )

        # try to load the pretrained model of CoAT
        if encoder_pretrain is not None:
            checkpoint = torch.load(encoder_pretrain, map_location=lambda storage, loc: storage)
            self.encoder.load_state_dict(checkpoint['model'], strict=False)

    def forward(self, x):
        x = self.rgb(x)
        encode_info = self.encoder(x)

        class_codes = self.class_encoder(encode_info[-1])
        class_ca = self.class_linear(class_codes)
        last, _ = self.decoder(encode_info)

        last = last * class_ca
        logit = self.logit(last)
        grads = self.grad_conv(last)

        out = torch.cat([logit, grads], dim=1)

        return out, class_codes
