#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted form MONAI Tutorial: https://github.com/Project-MONAI/tutorials/tree/main/2d_segmentation/torch
"""

import argparse
import os
import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

from skimage import measure, morphology

from transforms.utils import CellF1Metric, StainNormalized


def main():
    parser = argparse.ArgumentParser("Microscopy image segmentation")
    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="./data/Train_Pre_3class/",
        type=str,
        help="training data path; subfolders: images, labels",
    )
    parser.add_argument(
        "--work_dir", default="debug", help="path where to save models and logs"
    )
    parser.add_argument("--seed", default=2022, type=int)
    # parser.add_argument("--resume", default=False, help="resume from checkpoint")
    parser.add_argument("--num_workers", default=4, type=int)

    # Model parameters
    parser.add_argument(
        "--model_name", default="swinunetr", help="select mode: unet, unetr, swinunetr， swinunetr_dfc_v3"
    )
    parser.add_argument("--num_class", default=3, type=int, help="segmentation classes")
    parser.add_argument(
        "--input_size", default=256, type=int, help="segmentation classes"
    )
    parser.add_argument('--model_path', default='./work_dir/swinunetrv2_3class', help='path where to save models and segmentation results')

    # Training parameters
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU")
    parser.add_argument("--max_epochs", default=2000, type=int)
    parser.add_argument("--val_interval", default=2, type=int)
    parser.add_argument("--epoch_tolerance", default=150, type=int)
    parser.add_argument("--initial_lr", type=float, default=6e-4, help="learning rate")

    args = parser.parse_args()

    from model_selector import model_factory

    from transforms.utils import ConditionChannelNumberd
    from monai.utils import GridSampleMode

    join = os.path.join

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    import monai
    from monai.data import decollate_batch, PILReader
    from monai.inferers import sliding_window_inference
    from monai.metrics import DiceMetric
    from monai.transforms import (
        Activations,
        AddChanneld,
        AsDiscrete,
        Compose,
        LoadImaged,
        SpatialPadd,
        RandSpatialCropd,
        RandRotate90d,
        ScaleIntensityd,
        RandAxisFlipd,
        RandZoomd,
        RandGaussianNoised,
        RandAdjustContrastd,
        RandGaussianSmoothd,
        RandHistogramShiftd,
        EnsureTyped,
        EnsureType, EnsureChannelFirstd,
        Rand2DElasticd, GaussianSmooth
    )
    from monai.visualize import plot_2d_or_3d_image
    from datetime import datetime
    import shutil

    print("Successfully imported all requirements!")

    monai.config.print_config()

    # %% set training/validation split
    np.random.seed(args.seed)
    model_path = join(args.work_dir, args.model_name + "_3class")
    os.makedirs(model_path, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    shutil.copyfile(
        __file__, join(model_path, run_id + "_" + os.path.basename(__file__))
    )
    img_path = join(args.data_path, "images")
    gt_path = join(args.data_path, "labels")

    img_names = sorted(os.listdir(img_path))
    # modified to tiff
    gt_names = [img_name.split(".")[0] + "_label.png" for img_name in img_names]
    img_num = len(img_names)
    val_frac = 0.1
    indices = np.arange(img_num)
    np.random.shuffle(indices)
    val_split = int(img_num * val_frac)
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    train_files = [
        {"img": join(img_path, img_names[i]), "label": join(gt_path, gt_names[i])}
        for i in train_indices
    ]
    val_files = [
        {"img": join(img_path, img_names[i]), "label": join(gt_path, gt_names[i])}
        for i in val_indices
    ]
    print(
        f"training image num: {len(train_files)}, validation image num: {len(val_files)}"
    )
    # %% define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(
                keys=["img", "label"], reader=PILReader, dtype=np.uint8
            ),  # image three channels (H, W, 3); label: (H, W)
            AddChanneld(keys=["label"], allow_missing_keys=True),  # label: (1, H, W)
            # ConditionAddChannelLastd(
            #     keys=["img"], target_dims=2, allow_missing_keys=True
            # ),
            EnsureChannelFirstd(
                keys=["img"], strict_check=False
            ),  # image: (3, H, W)
            ConditionChannelNumberd(
                keys=["img"], target_dim=0, channel_num=3, allow_missing_keys=True
            ),
            ScaleIntensityd(
                keys=["img"], allow_missing_keys=True
            ),  # Do not scale label

            SpatialPadd(keys=["img", "label"], spatial_size=args.input_size),
            RandSpatialCropd(
                keys=["img", "label"], roi_size=args.input_size, random_size=False
            ),
            RandAxisFlipd(keys=["img", "label"], prob=0.5),
            RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
            # Rand2DElasticd(keys=["img", "label"], spacing=(7, 7), magnitude_range=(-3, 3), mode=[GridSampleMode.BILINEAR, GridSampleMode.NEAREST]),
            # # intensity transform
            RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
            RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
            RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2), sigma_y=(1, 2)),
            RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
            RandZoomd(
                keys=["img", "label"],
                prob=1,
                min_zoom=0.3,
                max_zoom=1.5,
                mode=["area", "nearest"],
                padding_mode="constant"
            ),
            EnsureTyped(keys=["img", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.uint8),
            AddChanneld(keys=["label"], allow_missing_keys=True),
            # ConditionAddChannelLastd(
            #     keys=["img"], target_dims=2, allow_missing_keys=True
            # ),
            EnsureChannelFirstd(
                keys=["img"],
            ),  # image: (3, H, W)
            ConditionChannelNumberd(
                keys=["img"], target_dim=0, channel_num=3, allow_missing_keys=True
            ),
            # AsChannelFirstd(keys=["img"], channel_dim=-1, allow_missing_keys=True),
            ScaleIntensityd(keys=["img"], allow_missing_keys=True),
            # AsDiscreted(keys=['label'], to_onehot=3),
            EnsureTyped(keys=["img", "label"]),
        ]
    )

    # % define dataset, data loader
    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=4)
    check_data = monai.utils.misc.first(check_loader)
    print(
        "sanity check:",
        check_data["img"].shape,
        torch.max(check_data["img"]),
        check_data["label"].shape,
        torch.max(check_data["label"]),
    )

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    dice_metric = DiceMetric(
        include_background=False, reduction="mean", get_not_nans=False
    )
    f1_metric = CellF1Metric(get_not_nans=False)

    post_pred = Compose(
        [EnsureType(), Activations(softmax=True)]
    )
    post_gt = Compose([EnsureType(), AsDiscrete(to_onehot=None)])
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_factory(args.model_name.lower(), device, args, in_channels=3)
    # from augment.stain_augment.StainNet.models import StainNet
    # stain_model = StainNet()
    # check_point = "./augment/stain_augment/StainNet/checkpoints/aligned_cytopathology_dataset/StainNet-3x0_best_psnr_layer3_ch32.pth"
    # stain_model.load_state_dict(torch.load(check_point))
    # stain_model.eval()
    # stain_model.requires_grad_(False)

    # find best model
    model_path = join(args.model_path, 'best_F1_model.pth')
    if not os.path.exists(model_path):
        model_path = join(args.model_path, 'best_Dice_model.pth')

    print(f"Loading {model_path}...")
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])

    initial_lr = args.initial_lr
    # smooth_transformer = GaussianSmooth(sigma=1)

    # start a typical PyTorch training
    max_epochs = args.max_epochs
    epoch_tolerance = args.epoch_tolerance
    val_interval = args.val_interval
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    torch.autograd.set_detect_anomaly(True)
    # writer = SummaryWriter(model_path)

    t = [0.5, 0.6, 0.7, 0.8, 0.9]
    res = {}

    for tho in t:
        model.eval()
        with torch.no_grad():

            for step, val_data in enumerate(val_loader, 1):
                val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
                # val_images = stain_model(val_images).to(device)

                val_labels_onehot = monai.networks.one_hot(
                    val_labels, args.num_class
                )
                roi_size = (args.input_size, args.input_size)
                sw_batch_size = args.batch_size

                val_outputs = sliding_window_inference(
                    val_images, roi_size, sw_batch_size, model
                )
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels_onehot = [
                    post_gt(i) for i in decollate_batch(val_labels_onehot)
                ]

                outputs_pred_npy = val_outputs[0][1].cpu().numpy()
                outputs_label_npy = val_labels_onehot[0][1].cpu().numpy()
                # convert probability map to binary mask and apply morphological postprocessing

                outputs_pred_mask = measure.label(morphology.remove_small_objects(morphology.remove_small_holes(outputs_pred_npy > tho), 16), connectivity=1)
                outputs_label_mask = measure.label(morphology.remove_small_objects(morphology.remove_small_holes(outputs_label_npy > 0.5), 16), connectivity=1)

                # convert back to tensor for metric computing
                outputs_pred_mask = torch.from_numpy(outputs_pred_mask[None, None])
                outputs_label_mask = torch.from_numpy(outputs_label_mask[None, None])

                f1 = f1_metric(y_pred=outputs_pred_mask, y=outputs_label_mask)
                dice = dice_metric(y_pred=val_outputs, y=val_labels_onehot)

                print(os.path.basename(
                    val_data["img_meta_dict"]["filename_or_obj"][0]
                ), f1, dice)

            # aggregate the final mean f1 score and dice result
            f1_metric_ = f1_metric.aggregate()[0].item()
            dice_metric_ = dice_metric.aggregate().item()

            # reset the status for next validation round
            dice_metric.reset()
            f1_metric.reset()
            metric_values.append(f1_metric_)

            # print(
            #     "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
            #         epoch + 1, dice_metric, best_metric, best_metric_epoch
            #     )
            # )
            print(
                "current mean f1 score: {:.4f}, mean dice: {:.4f} at t={}.".format(
                    f1_metric_, dice_metric_, tho
                )
            )

            res[tho] = f1_metric_
        # writer.add_scalars("val_metrics", {"f1": f1_metric_, "dice": dice_metric_}, 0)
    print(res)


if __name__ == "__main__":
    main()
