#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted form MONAI Tutorial: https://github.com/Project-MONAI/tutorials/tree/main/2d_segmentation/torch
"""

import argparse
import os

import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

from losses.sim import DirectionLoss
from transformers import flow_gen
from transformers.utils import CellF1Metric, dx_to_circ, TiffReader2, flow, fig2data, Flow2dTransposeFixd, Flow2dRoatation90Fixd, Flow2dFlipFixd
import monai.networks


def label2seg_and_grad(labels):
    seg_label = labels[:, :1]
    seg_label[seg_label > 0] = 1
    labels_onehot = monai.networks.one_hot(
        seg_label, 2
    )

    return labels_onehot, labels[:, 2:]


def output2seg_and_grad(outputs):
    return outputs[:, :2], outputs[:, 2:]


def main():
    parser = argparse.ArgumentParser("Microscopy image segmentation")
    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="./data/Train_Pre_grad/",
        type=str,
        help="training data path; subfolders: images, labels",
    )
    parser.add_argument(
        "--work_dir", default="debug", help="path where to save models and logs"
    )
    parser.add_argument("--seed", default=2022, type=int)
    # parser.add_argument("--resume", default=False, help="resume from checkpoint")
    parser.add_argument("--num_workers", default=8, type=int)

    # Model parameters
    parser.add_argument(
        "--model_name", default="coat_daformer_net_grad", help="select mode: coat_daformer_net_grad"
    )
    parser.add_argument("--num_class", default=4, type=int, help="segmentation classes")
    parser.add_argument(
        "--input_size", default=512, type=int, help="input size"
    )
    # Training parameters
    parser.add_argument("--batch_size", default=6, type=int, help="Batch size per GPU")
    parser.add_argument("--max_epochs", default=1000, type=int)
    parser.add_argument("--val_interval", default=2, type=int)
    parser.add_argument("--epoch_tolerance", default=100, type=int)
    parser.add_argument("--initial_lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--amp", type=bool, default=True, help="using amp")

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
    from monai.metrics import DiceMetric
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
    model_path = join(args.work_dir, args.model_name + "_grad")
    os.makedirs(model_path, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    shutil.copyfile(
        __file__, join(model_path, run_id + "_" + os.path.basename(__file__))
    )
    img_path = join(args.data_path, "images")
    gt_path = join(args.data_path, "labels")

    img_names = sorted(os.listdir(img_path))
    # modified to tiff
    gt_names = [img_name.split(".")[0] + "_label_flows.tif" for img_name in img_names]
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
                keys=["img"], reader=PILReader()
            ),  # image three channels (H, W, 3); label: (H, W)
            LoadImaged(keys=["label"], reader=TiffReader2(channel_dim=0)),
            Flow2dTransposeFixd(keys=["label"], flow_dim_start=2, flow_dim_end=4),

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
            # SpatialPadd(keys=["img", "label"], spatial_size=args.input_size),
            RandAxisFlipd(keys=["img", "label"], prob=0.5),
            Flow2dFlipFixd(keys=["label"], flow_dim_start=2, flow_dim_end=4),
            RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
            Flow2dRoatation90Fixd(keys=["label"], flow_dim_start=2, flow_dim_end=4),
            # Rand2DElasticd(keys=["img", "label"], spacing=(7, 7), magnitude_range=(-3, 3), mode=[GridSampleMode.BILINEAR, GridSampleMode.NEAREST]),
            # # intensity transform
            RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
            RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
            RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2), sigma_y=(1, 2)),
            RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
            RandZoomd(
                keys=["img", "label"],
                prob=0.5,
                min_zoom=0.5,
                max_zoom=2,
                mode=["area", "nearest"],
                padding_mode="constant"
            ),
            EnsureTyped(keys=["img", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["img"], reader=PILReader()),
            LoadImaged(keys=["label"], reader=TiffReader2(channel_dim=0)),
            Flow2dTransposeFixd(keys=["label"], flow_dim_start=2, flow_dim_end=4),
            # AddChanneld(keys=["label"], allow_missing_keys=True),
            # ConditionAddChannelLastd(
            #     keys=["img"], target_dims=2, allow_missing_keys=True
            # ),


            EnsureChannelFirstd(
                keys=["img"],
            ),  # image: (3, H, W)
            ConditionChannelNumberd(
                keys=["img"], target_dim=0, channel_num=3, allow_missing_keys=True
            ),
            # RandAxisFlipd(keys=["img", "label"], prob=1),
            # Flow2dFlipFixd(keys=["label"], flow_dim_start=2, flow_dim_end=4),
            # RandRotate90d(keys=["img", "label"], prob=1, spatial_axes=[0, 1]),
            # Flow2dRoatation90Fixd(keys=["label"], flow_dim_start=2, flow_dim_end=4),
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

    # post_pred = Compose(
    #     [EnsureType(), Activations(softmax=True), AsDiscrete(threshold=0.5)]
    # )
    # post_gt = Compose([EnsureType(), AsDiscrete(to_onehot=None)])
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_factory(args.model_name.lower(), device, args, in_channels=3)
    # from augment.stain_augment.StainNet.models import StainNet
    # stain_model = StainNet()
    # check_point = "./augment/stain_augment/StainNet/checkpoints/aligned_cytopathology_dataset/StainNet-3x0_best_psnr_layer3_ch32.pth"
    # stain_model.load_state_dict(torch.load(check_point))
    # stain_model.eval()
    # stain_model.requires_grad_(False)

    # loss_function = monai.losses.DiceCELoss(softmax=True).to(device)
    loss_function_1 = monai.losses.DiceCELoss(softmax=True).to(device)
    loss_function_2 = torch.nn.MSELoss()
    loss_function_3 = DirectionLoss()

    initial_lr = args.initial_lr
    optimizer = torch.optim.AdamW(model.parameters(), initial_lr)
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
    writer = SummaryWriter(model_path)

    amp = args.amp
    scaler = None
    if amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, max_epochs):
        model.train()
        epoch_loss = 0
        checkpoint = None

        train_bar = tqdm.tqdm(enumerate(train_loader, 1), total=len(train_loader))
        for step, batch_data in train_bar:
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(
                device
            )

            optimizer.zero_grad()
            # inputs = stain_model(inputs).to(device)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                labels_onehot, label_grad_yx = label2seg_and_grad(labels)
                pred_label, pred_grad = output2seg_and_grad(outputs)
                loss = loss_function_1(torch.softmax(pred_label, dim=1), labels_onehot) + \
                       5 * loss_function_2(pred_grad, label_grad_yx * 5) \
                        + loss_function_3.forward(pred_grad, label_grad_yx)

            if amp and scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
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
        if epoch >= 20 and epoch % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images_board = None
                val_labels_board = None
                val_outputs_board = None
                val_grad_board = None
                val_label_grad_board = None
                val_instance_label_board = None
                val_pred_instance_label_board = None

                for step, val_data in enumerate(val_loader, 1):
                    val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
                    # val_images = stain_model(val_images).to(device)

                    val_labels_onehot, label_grad = label2seg_and_grad(val_labels)

                    roi_size = (args.input_size, args.input_size)
                    sw_batch_size = args.batch_size

                    val_outputs = sliding_window_inference(
                        val_images, roi_size, sw_batch_size, model
                    )

                    pred_label, pred_grad = output2seg_and_grad(val_outputs)

                    pred_label = torch.softmax(pred_label, dim=1)

                    pred_grad_cpu = pred_grad.detach().cpu()[0]
                    pred_label_cpu = pred_label.detach().cpu()[0]
                    label_grad_cpu = label_grad.detach().cpu()[0]
                    val_grad = dx_to_circ(pred_grad_cpu).transpose(2, 0, 1)[None, ...]
                    val_label_grad = dx_to_circ(label_grad_cpu).transpose(2, 0, 1)[None, ...]

                    instance_label, _ = flow_gen.compute_masks(pred_grad_cpu.numpy(),
                                                               pred_label_cpu[1].numpy(),
                                                               cellprob_threshold=0.5, use_gpu=True, niter=400)

                    # numpy.ndarray -> torch.tensor
                    # H x W -> B x C x H x W
                    instance_label = torch.from_numpy(instance_label.astype(np.float32))[None, None, ...]
                    val_instance_label = val_labels[:, :1].cpu()

                    del pred_grad, label_grad

                    if step == epoch % len(val_loader):
                        g_step = max(pred_grad_cpu.shape[1] // 128, 1)
                        flow_ = pred_grad_cpu[:, ::g_step, ::g_step].numpy() / 2

                        fig, _, _ = flow([flow_.transpose(1, 2, 0)], show=False, width=10)
                        val_grad_board = val_grad
                        fig_tensor = fig2data(fig)[:, :, :3].transpose(2, 0, 1)[None, ...]

                        flow_ = label_grad_cpu[:, ::g_step, ::g_step].numpy() * 2
                        fig, _, _ = flow([flow_.transpose(1, 2, 0)], show=False, width=10)
                        # val_grad_board = val_grad
                        label_fig_tensor = fig2data(fig)[:, :, :3].transpose(2, 0, 1)[None, ...]

                        val_outputs = val_grad * (pred_label_cpu[1].numpy() > 0.5)
                        val_images_board = val_images
                        val_labels_board = val_labels
                        val_outputs_board = val_outputs
                        val_label_grad_board = val_label_grad
                        # val_instance_label_board = val_instance_label
                        # val_pred_instance_label_board = instance_label

                    f1 = f1_metric(y_pred=instance_label, y=val_instance_label)
                    print(instance_label.max(), val_instance_label.max())
                    dice = dice_metric(y_pred=pred_label, y=val_labels_onehot)

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
                if f1_metric_ > best_metric and checkpoint is not None:
                    best_metric = f1_metric_
                    best_metric_epoch = epoch + 1
                    # torch.save(checkpoint, join(model_path, "best_Dice_model.pth"))
                    torch.save(checkpoint, join(model_path, "best_F1_model.pth"))
                    print("saved new best metric model")
                # print(
                #     "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                #         epoch + 1, dice_metric, best_metric, best_metric_epoch
                #     )
                # )
                print(
                    "current epoch: {} current mean f1 score: {:.4f} best mean f1 score: {:.4f} at epoch {}".format(
                        epoch + 1, f1_metric_, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalars("val_metrics", {"f1": f1_metric_, "dice": dice_metric_}, epoch + 1)

                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images_board, epoch, writer, index=0, tag="image", max_channels=3)
                plot_2d_or_3d_image(val_labels_board, epoch, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs_board, epoch, writer, index=0, tag="output", max_channels=3)
                plot_2d_or_3d_image(val_grad_board, epoch, writer, index=0, tag="grad", max_channels=3)
                plot_2d_or_3d_image(fig_tensor, epoch, writer, index=0, tag="flow", max_channels=3)
                plot_2d_or_3d_image(label_fig_tensor, epoch, writer, index=0, tag="label_flow", max_channels=3)
                plot_2d_or_3d_image(val_label_grad_board, epoch, writer, index=0, tag="label_grad", max_channels=3)

                # plot_2d_or_3d_image(val_instance_label_board, epoch, writer, index=0, tag="label_instance")
                # plot_2d_or_3d_image(val_pred_instance_label_board, epoch, writer, index=0, tag="label_pred_instance")
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
