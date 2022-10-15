from numba import jit

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


class DaFormaerCoATNet_GRAD_V5(nn.Module):

    def __init__(self,
                 in_channel=3,
                 out_channel=4,
                 classes_num=4,
                 encoder=coat_lite_medium,
                 encoder_pretrain='coat_lite_medium_384x384_f9129688.pth',
                 decoder=daformer_involution,
                 decoder_dim=320,
                 dropout_p=0.5):
        super(DaFormaerCoATNet_GRAD_V5, self).__init__()

        assert out_channel > 3
        # channel0: in prob
        # channel1: out prob
        # channel2: grad y
        # channel3: grad x

        self.rgb = RGB()

        self.encoder = encoder(in_channels=in_channel)

        encoder_dims = self.encoder.embed_dims

        self.class_encoder = nn.Sequential(
            Conv2dBnReLU(encoder_dims[-1], encoder_dims[-1]),
            Conv2dBnReLU(encoder_dims[-1], encoder_dims[-1]),
            nn.AdaptiveAvgPool2d(1),
            # cell images has 4 modalities
            nn.Conv2d(in_channels=encoder_dims[-1], out_channels=classes_num, kernel_size=1),
            nn.BatchNorm2d(classes_num),
            nn.ReLU(inplace=True)
        )

        # shuffle the channels
        self.class_linear = nn.Linear(
            classes_num,
            classes_num
        )
        # self.class_dropout = nn.Dropout(inplace=True, p=dropout_p)

        self.decoder_0 = decoder(
            encoder_dim=encoder_dims,
            decoder_dim=decoder_dim // 2,
        )

        self.decoder_1 = decoder(
            encoder_dim=encoder_dims,
            decoder_dim=decoder_dim // 2,
        )

        self.decoder_2 = decoder(
            encoder_dim=encoder_dims,
            decoder_dim=decoder_dim // 2,
        )

        self.decoder_3 = decoder(
            encoder_dim=encoder_dims,
            decoder_dim=decoder_dim // 2,
        )

        # self.mlps = nn.ModuleList(
        #     [
        #         nn.Sequential(nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1),
        #                       nn.BatchNorm2d(decoder_dim),
        #                       ),
        #         nn.Sequential(nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1),
        #                       nn.BatchNorm2d(decoder_dim),
        #                       ),
        #         nn.Sequential(nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1),
        #                       nn.BatchNorm2d(decoder_dim),
        #                       ),
        #         nn.Sequential(nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1),
        #                       nn.BatchNorm2d(decoder_dim),
        #                       )
        #     ]
        # )

        self.logit = nn.Sequential(
            nn.Conv2d(decoder_dim // 2, decoder_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_dim),
            PixelShuffleBlock(decoder_dim, out_channel - 2, upscale_factor=4),
            nn.BatchNorm2d(out_channel - 2),
        )
        self.grad_conv = nn.Sequential(
            nn.Conv2d(decoder_dim // 2, decoder_dim, kernel_size=3, padding=1),
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

        self.output_latent = False
        self.class_sup = False

    def output_latent_enable(self, enable=True):
        self.output_latent = enable

    def forward(self, x, noise=0.0, class_sup=False):
        if class_sup or self.class_sup:
            return self._forward_with_class(x, noise)
        return self._forward(x, noise)

    def _forward(self, x, noise=0.0):
        encode_info = self.encode(x)
        # add noise for VAT
        encode_info[-1] = encode_info[-1] + noise
        out = self.decode(encode_info)

        if self.output_latent:
            return out, encode_info

        return out

    def _forward_with_class(self, x, noise=0.0):
        encode_info = self.encode(x)
        class_code, class_ca = self.class_encode(encode_info)

        # add noise for VAT
        encode_info[-1] = encode_info[-1] + noise
        confidence = torch.std(class_code, dim=1)
        out = self.class_ca_decode(encode_info, class_ca, confidence)

        if self.output_latent:
            return out, class_code, encode_info

        return out, class_code

    def encode(self, x):
        x = self.rgb(x)
        encode_info = self.encoder(x)
        return encode_info

    def decode(self, latent):
        last, decode_info = self.decoder_0(latent)

        logit = self.logit(last)
        grads = self.grad_conv(last)

        out = torch.cat([logit, grads], dim=1)
        return out

    def class_ca_decode(self, latent, class_ca, conf):

        select = F.gumbel_softmax(class_ca, dim=1, tau=100 * (conf.mean().item() - 0.5) ** 2 + 1e-5)
        class_num = class_ca.shape[1]
        last_ca = 0
        for i in range(class_num):
            last = getattr(self, f'decoder_{i}')(latent)[0]
            last_ca = last_ca + last * select[:, i:i + 1, None, None] / class_num

        logit = self.logit(last_ca)
        grads = self.grad_conv(last_ca)

        out = torch.cat([logit, grads], dim=1)
        return out

    def class_encode(self, encode_info):
        class_codes = self.class_encoder(encode_info[-1].detach())
        class_codes = torch.flatten(class_codes, 1, -1)
        class_ca = self.class_linear(class_codes.detach())
        return class_codes, class_ca
