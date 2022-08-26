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


class DaFormaerCoATNet(nn.Module):

    def __init__(self,
                 in_channel=3,
                 out_channel=3,
                 encoder=coat_lite_medium,
                 encoder_pretrain='coat_lite_medium_384x384_f9129688.pth',
                 decoder=daformer_conv3x3,
                 decoder_dim=320):
        super(DaFormaerCoATNet, self).__init__()

        self.rgb = RGB()

        self.encoder = encoder(in_channels=in_channel)

        # try to load the pretrained model of CoAT
        if encoder_pretrain is not None:
            checkpoint = torch.load(encoder_pretrain, map_location=lambda storage, loc: storage)
            self.encoder.load_state_dict(checkpoint['model'], strict=False)
        encoder_dim = self.encoder.embed_dims

        self.decoder = decoder(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
        )
        self.logit = nn.Sequential(
            nn.Conv2d(decoder_dim, out_channel, kernel_size=1),
        )

    def forward(self, x):

        x = self.rgb(x)
        encoder = self.encoder(x)

        last, decoder = self.decoder(encoder)

        logit = self.logit(last)

        upsample_logit = F.interpolate(logit, size=None, scale_factor=4, mode='bilinear', align_corners=False)

        return upsample_logit

