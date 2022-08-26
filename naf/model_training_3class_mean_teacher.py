#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted form MONAI Tutorial: https://github.com/Project-MONAI/tutorials/tree/main/2d_segmentation/torch
"""

import argparse
import copy
import os
import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from monai.transforms import allow_missing_keys_mode
import training.ramp as ramps

from skimage import measure, morphology

from transformers.utils import CellF1Metric


def update_ema_variables(model, ema_model, global_step, alpha=0.999):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def get_current_consistency_weight(epoch):
    # default parameters
    # --consistency 100.0
    # --consistency-rampup 5
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 100 * ramps.sigmoid_rampup(epoch, 5)


def main():
    parser = argparse.ArgumentParser("Microscopy image segmentation")
    # Dataset parameters
    parser.add_argument(
        "--labeled_path",
        default="./data/Train_Pre_3class/",
        type=str,
        help="training data path; subfolders: images, labels",
    )

    parser.add_argument(
        "--unlabeled_path",
        default="./data/Train_Unlabeled/",
        type=str,
        help="training unlabeled data path",
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
    # Training parameters
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU")
    parser.add_argument("--max_epochs", default=2000, type=int)
    parser.add_argument("--val_interval", default=2, type=int)
    parser.add_argument("--epoch_tolerance", default=150, type=int)
    parser.add_argument("--initial_lr", type=float, default=6e-4, help="learning rate")
    parser.add_argument("--alpha", type=float, default=0.999, help="ema")

    args = parser.parse_args()

    from model_selector import model_factory

    from transformers.utils import ConditionChannelNumberd

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
    model_path = join(args.work_dir, args.model_name + "_3class_mean_teacher")
    os.makedirs(model_path, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    shutil.copyfile(
        __file__, join(model_path, run_id + "_" + os.path.basename(__file__))
    )
    img_path = join(args.labeled_path, "images")
    unlabeled_img_path = args.unlabeled_path
    gt_path = join(args.labeled_path, "labels")

    labeled_img_names = sorted(os.listdir(img_path))
    unlabeled_img_names = sorted(os.listdir(unlabeled_img_path))

    gt_names = [img_name.split(".")[0] + "_label.png" for img_name in labeled_img_names]
    img_num = len(labeled_img_names)
    unlabeled_img_num = len(unlabeled_img_names)

    val_frac = 0.1

    val_split = int(img_num * val_frac)
    indices = np.arange(img_num)

    unlabeled_train_indices = np.arange(unlabeled_img_num)

    combined_img_num = img_num - val_split + unlabeled_img_num
    np.random.shuffle(indices)
    # combined_train_indices = np.arange(combined_img_num)

    val_indices = np.arange(val_split)
    train_combined_indices = np.arange(img_num - val_split)

    train_labeled_files = [
        {"img": join(img_path, labeled_img_names[i]), "label": join(gt_path, gt_names[i])}
        for i in train_combined_indices
    ]

    train_unlabeled_files = [
        {"img": join(unlabeled_img_path, unlabeled_img_names[i])}
        for i in unlabeled_train_indices
    ]

    # mixed_train_files = train_labeled_files + train_unlabeled_files

    # train_files = [
    #     mixed_train_files[i]
    #     for i in combined_train_indices
    # ]

    val_files = [
        {"img": join(img_path, labeled_img_names[i]), "label": join(gt_path, gt_names[i])}
        for i in val_indices
    ]
    print(
        f"training labeled image num: {len(train_labeled_files)}, unlabeled image num: {unlabeled_img_num}, validation image num: {len(val_files)}"
    )
    # %% define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(
                keys=["img", "label"], reader=PILReader, dtype=np.uint8, allow_missing_keys=True
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
            SpatialPadd(keys=["img", "label"], spatial_size=args.input_size, allow_missing_keys=True),
            RandSpatialCropd(
                keys=["img", "label"], roi_size=args.input_size, random_size=False, allow_missing_keys=True
            ),
            RandAxisFlipd(keys=["img", "label"], prob=0.5, allow_missing_keys=True),
            RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1], allow_missing_keys=True),
            # Rand2DElasticd(keys=["img", "label"], spacing=(7, 7), magnitude_range=(-3, 3), mode=[GridSampleMode.BILINEAR, GridSampleMode.NEAREST]),
            # # intensity transform
            RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1, allow_missing_keys=True),
            RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2), allow_missing_keys=True),
            RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2), sigma_y=(1, 2), allow_missing_keys=True),
            RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3, allow_missing_keys=True),
            RandZoomd(
                keys=["img", "label"],
                prob=1,
                min_zoom=0.3,
                max_zoom=2.5,
                mode=["area", "nearest"],
                padding_mode="constant", allow_missing_keys=True
            ),
            EnsureTyped(keys=["img", "label"], allow_missing_keys=True),
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

    mean_teacher_transforms = Compose(
        [
            RandAxisFlipd(keys=["img"], prob=1, allow_missing_keys=True),
            RandRotate90d(keys=["img"], prob=1, spatial_axes=(1, 2), allow_missing_keys=True),
            RandGaussianNoised(keys=["img"], prob=1, mean=0, std=0.1, allow_missing_keys=True),
            RandAdjustContrastd(keys=["img"], prob=1, gamma=(1, 2), allow_missing_keys=True),
            # EnsureTyped(keys=["img"], allow_missing_keys=True),
        ]
    )

    # % define dataset, data loader
    check_ds = monai.data.Dataset(data=train_labeled_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=4)
    check_data = monai.utils.misc.first(check_loader)
    print(
        "sanity check:",
        check_data["img"].shape,
        torch.max(check_data["img"]),
        # check_data["label"].shape,
        # torch.max(check_data["label"]),
    )

    # %% create a training data loader
    labeled_train_ds = monai.data.Dataset(data=labeled_img_names, transform=train_transforms)
    unlabeled_train_ds = monai.data.Dataset(data=unlabeled_img_names, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    labeled_train_loader = DataLoader(
        labeled_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    unlabeled_train_loader = DataLoader(
        labeled_train_ds,
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
    post_gt = Compose([EnsureType(), AsDiscrete(to_onehot=None)])
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student_model = model_factory(args.model_name.lower(), device, args, in_channels=3)
    teacher_model = copy.deepcopy(student_model)

    loss_function = monai.losses.DiceCELoss(softmax=True, ce_weight=torch.tensor([0.2, 0.3, 0.5]).to(device))
    consistency_function = torch.nn.MSELoss()

    initial_lr = args.initial_lr
    optimizer = torch.optim.AdamW(student_model.parameters(), initial_lr)

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

    alpha = args.alpha

    ########### mean teacher train process #############
    #  Double Streams applied at vanilla mean teacher
    #  paper, we mixed unlabeled and labeled images and
    #  the consistency loss will be calculated while the
    #  unlabeled images are sampled.
    ####################################################

    # keep eval
    teacher_model.eval()

    for epoch in range(1, max_epochs):
        student_model.train()
        epoch_loss = 0
        train_bar = tqdm.tqdm(enumerate(labeled_train_loader, 1), total=len(labeled_train_loader))
        train_bar2 = tqdm.tqdm(enumerate(unlabeled_train_loader, 1), total=len(unlabeled_train_loader))
        for (step, labeled_batch_data), (step, unlabeled_batch_data) in zip(train_bar, train_bar2):

            labels = labeled_batch_data["label"].to(device)
            inputs = labeled_batch_data["img"].to(device)

            optimizer.zero_grad()
            outputs = student_model(inputs)

            # applied augment for teacher model
            fixed_aug_inputs = mean_teacher_transforms({"img": inputs})
            infer_output = teacher_model(fixed_aug_inputs["img"].to(device)).detach()

            # inverse the transformers for segment results
            # fixed_aug_inputs.applied_operations = batch_data.applied_operations
            fixed_aug_inputs["img"] = infer_output
            with allow_missing_keys_mode(mean_teacher_transforms):
                back_t_outputs = mean_teacher_transforms.inverse(fixed_aug_inputs)

            # extract the label
            back_t_outputs = back_t_outputs["img"]

            # add a Consistency #Loss
            loss = consistency_function(outputs, back_t_outputs)

            labels_onehot = monai.networks.one_hot(
                labels, args.num_class
            )  # (b,cls,256,256)

            beta = get_current_consistency_weight(epoch)

            loss = loss * beta + loss_function(outputs, labels_onehot) * (1 - beta)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(labeled_train_ds) // labeled_train_loader.batch_size

            train_bar.set_postfix_str(f"train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch} average loss: {epoch_loss:.4f}")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": student_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss_values,
        }

        # update the teacher model at each epoch
        update_ema_variables(student_model, teacher_model, epoch, alpha=alpha)

        if epoch > 20 and epoch % val_interval == 0:
            student_model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                val_bar = tqdm.tqdm(enumerate(val_loader, 1), total=len(labeled_train_loader))

                for val_data in val_bar:
                    val_images, val_labels = val_data["img"].to(device), val_data[
                        "label"
                    ].to(device)
                    val_labels_onehot = monai.networks.one_hot(
                        val_labels, args.num_class
                    )
                    roi_size = (args.input_size, args.input_size)
                    sw_batch_size = args.batch_size

                    val_outputs = sliding_window_inference(
                        val_images, roi_size, sw_batch_size, student_model
                    )
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels_onehot = [
                        post_gt(i) for i in decollate_batch(val_labels_onehot)
                    ]

                    outputs_pred_npy = val_outputs[0][1].cpu().numpy()
                    outputs_label_npy = val_labels_onehot[0][1].cpu().numpy()
                    # convert probability map to binary mask and apply morphological postprocessing
                    outputs_pred_mask = measure.label(morphology.remove_small_objects(morphology.remove_small_holes(outputs_pred_npy > 0.5), 16))
                    outputs_label_mask = measure.label(morphology.remove_small_objects(morphology.remove_small_holes(outputs_label_npy > 0.5), 16))

                    # convert back to tensor for metric computing
                    outputs_pred_mask = torch.from_numpy(outputs_pred_mask[None, None])
                    outputs_label_mask = torch.from_numpy(outputs_label_mask[None, None])

                    f1 = f1_metric(y_pred=outputs_pred_mask, y=outputs_label_mask)
                    dice = dice_metric(y_pred=val_outputs, y=val_labels_onehot)

                    # compute metric for current iteration
                    # print(
                    #     os.path.basename(
                    #         val_data["img_meta_dict"]["filename_or_obj"][0]
                    #     ), f1, dice
                    # )

                    val_bar.set_postfix_str(os.path.basename(
                        val_data["img_meta_dict"]["filename_or_obj"][0]
                    ) + str(f1) + str(dice))

                # aggregate the final mean f1 score and dice result
                f1_metric_ = f1_metric.aggregate()[0].item()
                dice_metric_ = dice_metric.aggregate().item()
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
