from models.daformer import *
from models.coat import *


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


class DaFormaerCoATNet_GRAD(nn.Module):

    def __init__(self,
                 in_channel=3,
                 out_channel=4,
                 encoder=coat_lite_medium,
                 encoder_pretrain='coat_lite_medium_384x384_f9129688.pth',
                 decoder=daformer_conv3x3,
                 decoder_dim=320):
        super(DaFormaerCoATNet_GRAD, self).__init__()

        assert out_channel > 3
        # channel0: in prob
        # channel1: out prob
        # channel2: grad y
        # channel3: grad x

        self.rgb = RGB()

        self.encoder = encoder(in_channels=in_channel)

        encoder_dim = self.encoder.embed_dims

        self.decoder = decoder(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
        )
        self.logit = nn.Sequential(
            nn.Conv2d(decoder_dim, out_channel - 2, kernel_size=1),
        )
        self.grad_conv = nn.Sequential(
            nn.Conv2d(decoder_dim, 2, kernel_size=1),
        )

        # try to load the pretrained model of CoAT
        if encoder_pretrain is not None:
            checkpoint = torch.load(encoder_pretrain, map_location=lambda storage, loc: storage)
            self.encoder.load_state_dict(checkpoint['model'], strict=False)

    def forward(self, x):
        x = self.rgb(x)
        encode_info = self.encoder(x)

        last, decode_info = self.decoder(encode_info)

        logit = self.logit(last)
        grads = self.grad_conv(last)

        upsample_logit = F.interpolate(logit, size=None, scale_factor=4, mode='bilinear', align_corners=False)
        grads = F.interpolate(grads, size=None, scale_factor=4, mode='bilinear', align_corners=False)

        return torch.cat([upsample_logit, grads], dim=1)