#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted form MONAI Tutorial: https://github.com/Project-MONAI/tutorials/tree/main/2d_segmentation/torch
"""

import argparse
import os
import random

from losses import sim

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from scipy.ndimage import binary_dilation
from skimage.morphology import dilation

import skimage
from transformers.utils import CellF1Metric, sem2ins_label
from monai.transforms import NormalizeIntensityd

from losses.sim import MSEGrad2D


def main():
    parser = argparse.ArgumentParser("Baseline for Microscopy image segmentation")
    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="./data/Train_Pre_sdf/",
        type=str,
        help="training data path; subfolders: images, labels",
    )
    parser.add_argument(
        "--work_dir", default="./naf/work_dir/sdf", help="path where to save models and logs"
    )
    parser.add_argument("--seed", default=2022, type=int)
    # parser.add_argument("--resume", default=False, help="resume from checkpoint")
    parser.add_argument("--num_workers", default=4, type=int)

    # Model parameters
    parser.add_argument(
        "--model_name", default="swinunetr", help="select mode: unet, unetr, swinunetr， swinunetrv2"
    )
    parser.add_argument("--num_class", default=2, type=int, help="segmentation classes")
    parser.add_argument(
        "--input_size", default=256, type=int, help="segmentation classes"
    ),
    parser.add_argument('--model_path', default='./naf/work_dir/swinunetr_dfc_v3', help='path where to save models and segmentation results'),

    # Training parameters
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU")
    parser.add_argument("--max_epochs", default=2000, type=int)
    parser.add_argument("--val_interval", default=2, type=int)
    parser.add_argument("--epoch_tolerance", default=150, type=int)
    parser.add_argument("--initial_lr", type=float, default=6e-4, help="learning rate")

    args = parser.parse_args()

    from model_selector import model_factory

    from transformers.utils import ConditionChannelNumberd
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
    model_path = join(args.work_dir, args.model_name + "_sdf_fined")
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
        {"img": join(img_path, img_names[i]), "sdf_label": join(gt_path, gt_names[i])}
        for i in train_indices
    ]
    val_files = [
        {"img": join(img_path, img_names[i]), "sdf_label": join(gt_path, gt_names[i])}
        for i in val_indices
    ]
    print(
        f"training image num: {len(train_files)}, validation image num: {len(val_files)}"
    )
    # %% define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(
                keys=["img", "sdf_label"], reader=PILReader, dtype=np.uint8
            ),  # image three channels (H, W, 3); label: (H, W)
            AddChanneld(keys=["sdf_label"], allow_missing_keys=True),  # label: (1, H, W)
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
            SpatialPadd(keys=["img", "sdf_label"], spatial_size=args.input_size),
            RandSpatialCropd(
                keys=["img", "sdf_label"], roi_size=args.input_size, random_size=False
            ),
            ScaleIntensityd(keys=["sdf_label"], allow_missing_keys=True, minv=None, maxv=None, factor=(1 / 255 - 1)),
            RandAxisFlipd(keys=["img", "sdf_label"], prob=0.5),
            RandRotate90d(keys=["img", "sdf_label"], prob=0.5, spatial_axes=[0, 1]),
            Rand2DElasticd(keys=["img", "sdf_label"], spacing=(7, 7), magnitude_range=(-3, 3), mode=[GridSampleMode.BILINEAR, GridSampleMode.NEAREST]),
            # # intensity transform
            RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
            RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
            RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
            RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
            RandZoomd(
                keys=["img", "sdf_label"],
                prob=0.15,
                min_zoom=0.4,
                max_zoom=1.5,
                mode=["area", "nearest"],
            ),
            EnsureTyped(keys=["img", "sdf_label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "sdf_label"], reader=PILReader, dtype=np.uint8),
            AddChanneld(keys=["sdf_label"], allow_missing_keys=True),
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
            ScaleIntensityd(keys=["sdf_label"], allow_missing_keys=True, minv=None, maxv=None, factor=(1 / 255 - 1)),
            EnsureTyped(keys=["img", "sdf_label"]),
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
        check_data["sdf_label"].shape,
        torch.max(check_data["sdf_label"]),
    )

    # %% create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    dice_metric = DiceMetric(
        include_background=False, reduction="mean", get_not_nans=False
    )
    f1_metric = CellF1Metric(get_not_nans=False)

    post_pred = Compose(
        [EnsureType(), Activations(softmax=True), AsDiscrete(threshold=0.5)]
    )
    post_gt = Compose([EnsureType(), AsDiscrete(threshold=0.5)])
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_factory(args.model_name.lower(), device, args, in_channels=4)

    # loss_function = monai.losses.DiceFocalLoss(softmax=True).to(device)
    loss_function1 = monai.losses.DiceCELoss(softmax=True).to(device)
    loss_function2 = sim.LovaszSoftmaxLoss().to(device)

    # loss_function = monai.losses.DiceCELoss(softmax=True, ce_weight=torch.tensor([0.25, 0.25, 0.5]).to(device))
    # loss_function = monai.losses.GeneralizedDiceLoss()
    initial_lr = args.initial_lr
    optimizer = torch.optim.AdamW(model.parameters(), initial_lr)
    # smooth_transformer = GaussianSmooth(sigma=1)
    load_model_path = join(args.model_path, 'best_F1_model.pth')
    if not os.path.exists(load_model_path):
        load_model_path = join(args.model_path, 'best_Dice_model.pth')
    # load model parameters
    checkpoint = torch.load(load_model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), initial_lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer.param_groups[0]['capturable'] = True
    restart_epoch = checkpoint['epoch']
    history_loss = checkpoint['loss']

    # start a typical PyTorch training
    max_epochs = args.max_epochs
    epoch_tolerance = args.epoch_tolerance
    val_interval = args.val_interval
    best_metric = -1
    best_metric_epoch = restart_epoch
    epoch_loss_values = history_loss
    metric_values = list()
    torch.autograd.set_detect_anomaly(True)
    writer = SummaryWriter(model_path)

    print(f"restart from {restart_epoch} epoch...")
    for epoch in range(restart_epoch, max_epochs):
        model.train()
        epoch_loss = 0
        for step, batch_data in enumerate(train_loader, 1):
            inputs, labels = batch_data["img"].to(device), batch_data["sdf_label"].to(
                device
            )
            s = random.randint(0, 4)

            de = torch.ones_like(inputs[:, 0:1, ...], device=device, dtype=torch.float32) * s
            inputs = torch.cat([inputs, de], dim=1)

            optimizer.zero_grad()
            outputs = model(inputs)

            # labels[:, 2] = 1  # move the edge mask flag to inside mask flag
            # labels_onehot = monai.networks.one_hot(
            #     labels, args.num_class - 1
            # )  # (b,cls,256,256)

            in_out_label = labels.clone()

            out_bool = in_out_label > 0.5 - s / 256

            in_bool = torch.logical_not(out_bool)
            in_out_label[out_bool] = 0
            in_out_label[in_bool] = 1

            # print(in_out_label.max(),in_out_label.min())
            labels_onehot = monai.networks.one_hot(
                in_out_label, 2
            )  # (b,cls,256,256)

            loss = 0.8 * loss_function1(outputs, labels_onehot) + \
                    0.2 * loss_function2(outputs, labels_onehot)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step - 1}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch} average loss: {epoch_loss:.4f}")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss_values,
        }

        if epoch > 0 and epoch % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                in_out_label = None
                display_image = None
                display_label = None
                display_output = None
                display_inout_label = None
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(device), val_data[
                        "sdf_label"
                    ].to(device)

                    de = torch.zeros_like(val_images[:, 0:1, ...], device=device, dtype=torch.float32) + 2
                    val_images = torch.cat([val_images, de], dim=1)

                    roi_size = (256, 256)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(
                        val_images, roi_size, sw_batch_size, model
                    )

                    # sdf channel
                    # inside and outside
                    in_out_output = val_outputs

                    in_out_label = val_labels.clone()

                    out_bool = in_out_label > 0.5 - 1 / 256
                    in_bool = torch.logical_not(out_bool)
                    in_out_label[out_bool] = 0
                    in_out_label[in_bool] = 1
                    in_out_label = monai.networks.one_hot(
                        in_out_label, args.num_class
                    )
                    val_outputs_post = [post_pred(i) for i in decollate_batch(in_out_output)]
                    val_labels_post = [
                        post_gt(i) for i in decollate_batch(in_out_label)
                    ]

                    val_outputs_, val_labels_ = sem2ins_label(val_outputs_post, val_labels_post)

                    val_outputs_ = torch.from_numpy(dilation(val_outputs_.numpy()))
                    f1_metric_ = f1_metric(y_pred=val_outputs_, y=val_labels_)
                    dice_metric_ = dice_metric(y_pred=val_outputs_post, y=val_labels_post)
                    val_outputs = val_outputs_post[0]
                    display_inout_label = in_out_label
                    pic_name = os.path.basename(
                        val_data["img_meta_dict"]["filename_or_obj"][0]
                    )
                    # if '115' in pic_name:
                    display_image = val_images
                    display_label = val_labels
                    display_output = val_outputs

                    # compute metric for current iteration
                    print(
                        pic_name, dice_metric_, f1_metric_

                    )

                # aggregate the final mean dice result
                dice_metric_ = dice_metric.aggregate().item()
                f1_metric_ = f1_metric.aggregate()[0].item()
                # reset the status for next validation round
                dice_metric.reset()
                f1_metric.reset()
                metric_values.append(f1_metric_)
                if f1_metric_ > best_metric:
                    best_metric = f1_metric_
                    best_metric_epoch = epoch + 1
                    # torch.save(checkpoint, join(model_path, "best_Dice_model.pth"))
                    torch.save(checkpoint, join(model_path, "best_F1_model.pth"))
                    print("saved new best metric model")

                print(
                    "current epoch: {} current f1 score: {:.4f} best f1 score: {:.4f} at epoch {}".format(
                        epoch + 1, f1_metric_, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalars("val_metrics", {"mean_dice": dice_metric_, "f1_score": f1_metric_}, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(display_image, epoch, writer, index=0, tag="image", max_channels=3)
                plot_2d_or_3d_image(display_label, epoch, writer, index=0, tag="sdf_label")
                plot_2d_or_3d_image(display_inout_label, epoch, writer, index=0, tag="label")
                plot_2d_or_3d_image(display_output, epoch, writer, index=0, tag="output")
            if (epoch - best_metric_epoch) > epoch_tolerance:
                print(
                    f"validation metric does not improve for {epoch_tolerance} epochs! current {epoch=}, {best_metric_epoch=}"
                )
                break

    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )
    writer.close()
    torch.save(checkpoint, join(model_path, "final_model.pth"))
    np.savez_compressed(
        join(model_path, "train_log.npz"),
        val_dice=metric_values,
        epoch_loss=epoch_loss_values,
    )


if __name__ == "__main__":
    main()
