import torch
from torch import nn

from models import mpvit
from models.daformer import daformer_conv3x3


class DaFormaerMPVitNet(nn.Module):

    def __init__(self,
                 in_channel=3,
                 out_channel=3,
                 encoder=mpvit,
                 encoder_pretrain='coat_lite_medium_384x384_f9129688.pth',
                 decoder=daformer_conv3x3,
                 decoder_dim=320):
        super(DaFormaerMPVitNet, self).__init__()

        self.encoder = encoder(in_channels=in_channel)

        encoder_dim = self.encoder.embed_dims

        self.decoder = decoder(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
        )
        self.logit = nn.Sequential(
            nn.Conv2d(decoder_dim, out_channel, kernel_size=1),
        )

        # try to load the pretrained model of MPVit
        if encoder_pretrain is not None:
            checkpoint = torch.load(encoder_pretrain, map_location=lambda storage, loc: storage)
            self.encoder.load_state_dict(checkpoint['model'], strict=False)

    def forward(self, x):
        # x = self.rgb(x)
        encode_info = self.encoder(x)

        last, decode_info = self.decoder(encode_info)

        logit = self.logit(last)

        upsample_logit = F.interpolate(logit, size=None, scale_factor=4, mode='bilinear', align_corners=False)

        return upsample_logit