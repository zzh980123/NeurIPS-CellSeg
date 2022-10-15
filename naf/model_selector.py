import monai
from monai.networks.nets import SwinUNETR

from models import UNETR2D
from models.daformer_coat_net import DaFormaerCoATNet
from models.daformer_coat_net_center import DaFormaerCoATNet_Center
from models.daformer_coat_net_doubel_branch import DaFormaerCoATNet_db
from models.daformer_coat_net_grad import DaFormaerCoATNet_GRAD
from models.daformer_coat_net_grad_v2 import DaFormaerCoATNet_GRAD_V2
from models.daformer_coat_net_grad_v3 import DaFormaerCoATNet_GRAD_V3
from models.daformer_coat_net_grad_v3p1 import DaFormaerCoATNet_GRAD_V3p1
from models.daformer_coat_net_grad_v5 import DaFormaerCoATNet_GRAD_V5

from models.daformer_coat_net_v2 import DaFormaerCoATNet_v2
from models.daformer_coat_net_v3 import DaFormaerCoATNet_v3
from models.daformer_coat_net_v3_1 import DaFormaerCoATNet_v3_1
from models.swinunetr_dfc_v1 import SwinUNETRV2_DFCv1
from models.swinunetr_dfc_v2 import SwinUNETR_DFCv2
from models.swinunetr_dfc_v3 import SwinUNETR_DFCv3
from models.swinunetr_dfc_v4 import SwinUNETR_DFCv4
from models.swinunetr_dfc_v5 import SwinUNETR_DFCv5
from models.swinunetr_style import SwinUNETRStyle
from models.swinunetrv2 import SwinUNETRV2
from models.swinunetrv3 import SwinUNETRV3


def model_factory(model_name: str, device, args, in_channels=3, spatial_dims=2, pretrained_model_paths=[]):
    if model_name == 'unet':
        model = monai.networks.nets.UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=args.num_class,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)

    if model_name == 'unetr':
        assert spatial_dims == 2
        model = UNETR2D(
            in_channels=in_channels,
            out_channels=args.num_class,
            img_size=(args.input_size, args.input_size),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        ).to(device)

    if model_name == 'swinunetr':
        model = SwinUNETR(
            img_size=(args.input_size, args.input_size),
            in_channels=in_channels,
            out_channels=args.num_class,
            feature_size=24,  # should be divisible by 12
            spatial_dims=spatial_dims
        ).to(device)
    if model_name == 'swinunetr_emb24_2262':
        model = SwinUNETR(
            img_size=(args.input_size, args.input_size),
            in_channels=in_channels,
            out_channels=args.num_class,
            depths=(2, 2, 6, 2),
            feature_size=24,  # should be divisible by 12
            spatial_dims=spatial_dims
        ).to(device)

    if model_name == 'swinunetr_emb48_2262':
        model = SwinUNETR(
            img_size=(args.input_size, args.input_size),
            in_channels=in_channels,
            out_channels=args.num_class,
            feature_size=48,  # should be divisible by 12
            spatial_dims=spatial_dims,
            depths=(2, 2, 6, 2),
        ).to(device)

    if model_name == "swinunetrv2":
        model = SwinUNETRV2(
            img_size=(args.input_size, args.input_size),
            in_channels=in_channels,
            out_channels=args.num_class,
            feature_size=24,  # should be divisible by 12
            spatial_dims=spatial_dims,
        ).to(device)
    if model_name == "swinunetrv3":
        model = SwinUNETRV3(
            img_size=(args.input_size, args.input_size),
            in_channels=in_channels,
            norm_name="batch",
            out_channels=args.num_class,
            feature_size=48,  # should be divisible by 12
            spatial_dims=spatial_dims,
            depths=(2, 2, 6, 2)
        ).to(device)

    if model_name == "swinunetrv2_dfc":
        model = SwinUNETRV2_DFCv1(
            img_size=(args.input_size, args.input_size),
            in_channels=in_channels,
            # norm_name="batch",
            out_channels=args.num_class,
            feature_size=24,  # should be divisible by 12
            spatial_dims=spatial_dims,
        ).to(device)
    if model_name == "swinunetr_dfc_v2":
        model = SwinUNETR_DFCv2(
            img_size=(args.input_size, args.input_size),
            in_channels=in_channels,
            # norm_name="batch",
            out_channels=args.num_class,
            feature_size=24,  # should be divisible by 12
            spatial_dims=spatial_dims,
        ).to(device)
    if model_name == "swinunetr_dfc_v3":
        model = SwinUNETR_DFCv3(
            img_size=(args.input_size, args.input_size),
            in_channels=in_channels,
            # norm_name="batch",
            out_channels=args.num_class,
            feature_size=24,  # should be divisible by 12
            spatial_dims=spatial_dims,
        ).to(device)
    if model_name == "swinunetrstyle":
        model = SwinUNETRStyle(
            img_size=(args.input_size, args.input_size),
            in_channels=in_channels,
            # norm_name="batch",
            out_channels=args.num_class,
            feature_size=24,  # should be divisible by 12
            spatial_dims=spatial_dims,
        ).to(device)
    if model_name == "swinunetrstyle_emb48_2262":
        model = SwinUNETRStyle(
            img_size=(args.input_size, args.input_size),
            in_channels=in_channels,
            # norm_name="batch",
            out_channels=args.num_class,
            feature_size=48,  # should be divisible by 12
            spatial_dims=spatial_dims,
        ).to(device)
    if model_name == "swinunetr_dfc_v4":
        model = SwinUNETR_DFCv4(
            img_size=(args.input_size, args.input_size),
            in_channels=in_channels,
            # norm_name="batch",
            out_channels=args.num_class,
            feature_size=24,  # should be divisible by 12
            spatial_dims=spatial_dims,
        ).to(device)
    if model_name == "swinunetr_dfc_v5":
        model = SwinUNETR_DFCv5(
            img_size=(args.input_size, args.input_size),
            in_channels=in_channels,
            # norm_name="batch",
            out_channels=args.num_class,
            feature_size=24,  # should be divisible by 12
            spatial_dims=spatial_dims,
        ).to(device)
    if model_name == "swinunetr_dfc_v4_emb48_2262":
        model = SwinUNETR_DFCv4(
            img_size=(args.input_size, args.input_size),
            in_channels=in_channels,
            # norm_name="batch",
            out_channels=args.num_class,
            feature_size=48,  # should be divisible by 12
            spatial_dims=spatial_dims,
            depths=(2, 2, 6, 2)
        ).to(device)
    if model_name == "coat_daformer_net":
        model = DaFormaerCoATNet(
            in_channel=in_channels,
            out_channel=args.num_class,
            encoder_pretrain='naf/models/PretrainedModel/CoAT/coat_lite_medium_384x384_f9129688.pth'
        ).to(device)
    if model_name == "coat_daformer_net_v2":
        model = DaFormaerCoATNet_v2(
            in_channel=in_channels,
            out_channel=args.num_class,
            encoder_pretrain='naf/models/PretrainedModel/CoAT/coat_lite_medium_384x384_f9129688.pth'
        ).to(device)
    if model_name == "coat_daformer_net_v3":
        model = DaFormaerCoATNet_v3(
            in_channel=in_channels,
            out_channel=args.num_class,
            encoder_pretrain='naf/models/PretrainedModel/CoAT/coat_lite_medium_384x384_f9129688.pth'
        ).to(device)


    if model_name == "coat_daformer_net_v3_1":
        model = DaFormaerCoATNet_v3_1(
            in_channel=in_channels,
            out_channel=args.num_class,
            encoder_pretrain='naf/models/PretrainedModel/CoAT/coat_lite_medium_384x384_f9129688.pth'
        ).to(device)
    if model_name == "coat_daformer_net_db":
        model = DaFormaerCoATNet_db(
            in_channel=in_channels,
            out_channel=args.num_class,
            encoder_pretrain='naf/models/PretrainedModel/CoAT/coat_lite_medium_384x384_f9129688.pth'
        ).to(device)
    if model_name == "coat_daformer_net_grad":
        model = DaFormaerCoATNet_GRAD(
            in_channel=in_channels,
            out_channel=args.num_class,
            encoder_pretrain='naf/models/PretrainedModel/CoAT/coat_lite_medium_384x384_f9129688.pth'
        ).to(device)
    if model_name == "coat_daformer_net_grad_v2":
        model = DaFormaerCoATNet_GRAD_V2(
            in_channel=in_channels,
            out_channel=args.num_class,
            encoder_pretrain='naf/models/PretrainedModel/CoAT/coat_lite_medium_384x384_f9129688.pth'
        ).to(device)
    if model_name == "coat_daformer_net_grad_v3":
        model = DaFormaerCoATNet_GRAD_V3(
            in_channel=in_channels,
            out_channel=args.num_class,
            encoder_pretrain='naf/models/PretrainedModel/CoAT/coat_lite_medium_384x384_f9129688.pth'
        ).to(device)
    if model_name == "coat_daformer_net_grad_v3p1":
        model = DaFormaerCoATNet_GRAD_V3p1(
            in_channel=in_channels,
            out_channel=args.num_class,
            encoder_pretrain='naf/models/PretrainedModel/CoAT/coat_lite_medium_384x384_f9129688.pth'
        ).to(device)
    if model_name == "coat_daformer_net_grad_v5":
        model = DaFormaerCoATNet_GRAD_V5(
            in_channel=in_channels,
            out_channel=args.num_class,
            encoder_pretrain='naf/models/PretrainedModel/CoAT/coat_lite_medium_384x384_f9129688.pth'
        ).to(device)
    if model_name == "coat_daformer_net_center":
        model = DaFormaerCoATNet_Center(
            in_channel=in_channels,
            out_channel=args.num_class,
            encoder_pretrain='naf/models/PretrainedModel/CoAT/coat_lite_medium_384x384_f9129688.pth'
        ).to(device)
    return model
