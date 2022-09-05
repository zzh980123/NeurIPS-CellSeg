#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted form MONAI Tutorial: https://github.com/Project-MONAI/tutorials/tree/main/2d_segmentation/torch
"""

import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tqdm
from monai.metrics import MSEMetric

from transformers.utils import LoadJson2Tensor


def main():
    parser = argparse.ArgumentParser("Microscopy image size and style predict")
    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="./data/Train_Pre_cell_size/",
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
        "--model_name", default="coat_daformer_net_db", help="select mode: daformer_coat_net_v2_db"
    )
    parser.add_argument('--model_path', default='./naf/work_dir/coat_daformer_3class_s512_fined/coat_daformer_net_3class_fined',
                        help='path where to save models and segmentation results')

    parser.add_argument("--num_class", default=3, type=int, help="segmentation classes")
    parser.add_argument(
        "--input_size", default=512, type=int, help="segmentation classes"
    )
    # Training parameters
    parser.add_argument("--batch_size", default=6, type=int, help="Batch size per GPU")
    parser.add_argument("--max_epochs", default=2000, type=int)
    parser.add_argument("--val_interval", default=2, type=int)
    parser.add_argument("--epoch_tolerance", default=150, type=int)
    parser.add_argument("--initial_lr", type=float, default=1e-3, help="learning rate")

    args = parser.parse_args()

    from model_selector import model_factory

    from transformers.utils import ConditionChannelNumberd

    join = os.path.join

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    import monai
    from monai.data import PILReader
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        Activations,
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
        EnsureType, EnsureChannelFirstd
    )
    from monai.visualize import plot_2d_or_3d_image
    from datetime import datetime
    import shutil

    print("Successfully imported all requirements!")

    monai.config.print_config()

    # %% set training/validation split
    np.random.seed(args.seed)
    model_path = join(args.work_dir, args.model_name + "_size_style")
    os.makedirs(model_path, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    shutil.copyfile(
        __file__, join(model_path, run_id + "_" + os.path.basename(__file__))
    )
    img_path = join(args.data_path, "images")
    gt_path = join(args.data_path, "labels")

    img_names = sorted(os.listdir(img_path))
    # modified to tiff
    gt_names = [img_name.split(".")[0] + "_label.json" for img_name in img_names]
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
                keys=["img"], reader=PILReader, dtype=np.uint8
            ),  # image three channels (H, W, 3); label: (H, W)
            LoadJson2Tensor(keys=["label"], func=lambda obj: torch.tensor(obj)),
            EnsureChannelFirstd(
                keys=["img"], strict_check=False
            ),  # image: (3, H, W)
            ConditionChannelNumberd(
                keys=["img"], target_dim=0, channel_num=3, allow_missing_keys=True
            ),
            ScaleIntensityd(
                keys=["img"], allow_missing_keys=True
            ),  # Do not scale label
            SpatialPadd(keys=["img"], spatial_size=args.input_size),
            RandSpatialCropd(
                keys=["img"], roi_size=args.input_size, random_size=False
            ),
            RandAxisFlipd(keys=["img"], prob=0.5),
            RandRotate90d(keys=["img"], prob=0.5, spatial_axes=[0, 1]),
            # Rand2DElasticd(keys=["img", "label"], spacing=(7, 7), magnitude_range=(-3, 3), mode=[GridSampleMode.BILINEAR, GridSampleMode.NEAREST]),
            # # intensity transform
            RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
            RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
            RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2), sigma_y=(1, 2)),
            RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
            RandZoomd(
                keys=["img"],
                prob=1,
                min_zoom=0.2,
                max_zoom=5,
                mode=["area"],
                padding_mode="constant"
            ),
            EnsureTyped(keys=["img", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["img"], reader=PILReader, dtype=np.uint8),
            LoadJson2Tensor(keys=["label"], func=lambda obj: torch.tensor(obj)),
            EnsureChannelFirstd(
                keys=["img"],
            ),  # image: (3, H, W)
            ConditionChannelNumberd(
                keys=["img"], target_dim=0, channel_num=3, allow_missing_keys=True
            ),
            # AsChannelFirstd(keys=["img"], channel_dim=-1, allow_missing_keys=True),
            ScaleIntensityd(keys=["img"], allow_missing_keys=True),
            # AsDiscreted(keys=['label'], to_onehot=3),
            SpatialPadd(keys=["img"], spatial_size=args.input_size),
            RandSpatialCropd(
                keys=["img"], roi_size=args.input_size, random_size=False
            ),
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

    # dice_metric = DiceMetric(
    #     include_background=False, reduction="mean", get_not_nans=False
    # )
    # f1_metric = CellF1Metric(get_not_nans=False)

    mse_metric = MSEMetric()

    post_pred = Compose(
        [EnsureType(), Activations(softmax=True), AsDiscrete(threshold=0.5)]
    )
    post_gt = Compose([EnsureType(), AsDiscrete(to_onehot=None)])
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load segment part of the model
    model = model_factory(args.model_name.lower(), device, args, in_channels=3)
    load_model_path = join(args.model_path, 'best_F1_model.pth')
    checkpoint = torch.load(load_model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # loss_function = monai.losses.DiceCELoss(softmax=True).to(device)
    # loss_function = monai.losses.DiceCELoss(softmax=True, ce_weight=torch.tensor([0.2, 0.3, 0.5]).to(device))
    loss_function = torch.nn.MSELoss()
    initial_lr = args.initial_lr
    optimizer = torch.optim.AdamW(model.parameters(), initial_lr)
    # smooth_transformer = GaussianSmooth(sigma=1)

    # start a typical PyTorch training
    max_epochs = args.max_epochs
    epoch_tolerance = args.epoch_tolerance
    val_interval = args.val_interval
    best_metric = 1e12
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    torch.autograd.set_detect_anomaly(True)
    writer = SummaryWriter(model_path)
    for epoch in range(1, max_epochs):
        model.train()
        epoch_loss = 0
        train_bar = tqdm.tqdm(enumerate(train_loader, 1), total=len(train_loader))
        for step, batch_data in train_bar:
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(
                device
            )
            # print(batch_data)

            zooms = batch_data['img_transforms'][4]["extra_info"]["zoom"]
            zoom_batch = zooms[0].float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # labels_onehot = monai.networks.one_hot(
            #     labels, args.num_class
            # )  # (b,cls,256,256)

            # smooth edge
            # labels_onehot[:, 2, ...] = smooth_transformer(labels_onehot[:, 2, ...])
            pred_size = torch.sigmoid(outputs[:, 0]) * args.input_size ** 2

            zoomed_label = labels * zoom_batch * zoom_batch
            rate = pred_size / zoomed_label
            loss = loss_function(rate + 1 / rate - 2, torch.zeros_like(pred_size).to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size

            train_bar.set_postfix_str(f"train_loss: {loss.item():.4f}")
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

        if epoch > 20 and epoch % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None

                for step, val_data in enumerate(val_loader, 1):
                    val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
                    # val_labels_onehot = monai.networks.one_hot(
                    #     val_labels, args.num_class
                    # )

                    # roi_size = (args.input_size, args.input_size)
                    # sw_batch_size = args.batch_size

                    val_outputs = model(
                        val_images
                    )
                    # val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    # val_labels_onehot = [
                    #     post_gt(i) for i in decollate_batch(val_labels_onehot)
                    # ]

                    pred_size = torch.sigmoid(val_outputs[:, 0:1]) * args.input_size ** 2 + 1e-5
                    rate = pred_size / val_labels
                    metric = mse_metric(rate + 1 / rate - 2, torch.zeros_like(pred_size).to(device))

                    print(os.path.basename(
                        val_data["img_meta_dict"]["filename_or_obj"][0]
                    ), metric)

                # aggregate the final mean f1 score and dice result
                mse = mse_metric.aggregate().item()
                # reset the status for next validation round
                mse_metric.reset()

                metric_values.append(mse)
                if mse < best_metric:
                    best_metric = mse
                    best_metric_epoch = epoch + 1
                    # torch.save(checkpoint, join(model_path, "best_Dice_model.pth"))
                    torch.save(checkpoint, join(model_path, "best_MSE_model.pth"))
                    print("saved new best metric model")
                # print(
                #     "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                #         epoch + 1, dice_metric, best_metric, best_metric_epoch
                #     )
                # )
                print(
                    "current epoch: {} current mean mse: {:.4f} best mean mse: {:.4f} at epoch {}".format(
                        epoch + 1, mse, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalars("val_metrics", {"mse": mse}, epoch + 1)

                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch, writer, index=0, tag="output", max_channels=3)
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
