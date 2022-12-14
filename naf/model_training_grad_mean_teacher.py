#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted form MONAI Tutorial: https://github.com/Project-MONAI/tutorials/tree/main/2d_segmentation/torch
"""

import argparse
import copy
import itertools
import os
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import monai
import tqdm
import numpy as np

# not used
from monai.utils import GridSamplePadMode, GridSampleMode

from losses import sim
from training.datasets.mt_datasets import DualStreamDataset
from transforms import flow_gen

from monai.transforms import RandScaleIntensityd, RandRotated, RandRotate90d
import training.ramp as ramps

from transforms.utils import CellF1Metric, ColorJitterd, dx_to_circ, flow, fig2data, Flow2dTransposeFixd, TiffReader2, Flow2dRoatateFixd, Flow2dFlipFixd, RandCutoutd, \
    Flow2dRoatation90Fixd


def label2seg_and_grad(labels):
    seg_label = labels[:, 4:5]
    labels_onehot = monai.networks.one_hot(
        seg_label, 2
    )

    return labels[:, :1], labels_onehot, labels[:, 2:4]


def label2seg_and_grad2(labels):
    return labels[:, 4:5], labels[:, 2:4]


def output2seg_and_grad(outputs):
    return outputs[:, :2], outputs[:, 2:4]


# Fixed mean teacher model only update parameters but not buffers.
# Note: the buffers of the torch model may contains some other data, for example, the mask
# of the model parameters (model prune), if the model has some special data in the buffers,
# TRY TO STOP the ema.
def update_ema_variables(model, ema_model, global_step, alpha=0.999, use_buffers=True):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)

    def avg_fn(averaged_model_parameter, model_parameter):
        return averaged_model_parameter * alpha + model_parameter * (1 - alpha)

    # models with BN may need the buffers...
    student_param = (
        itertools.chain(model.parameters(), model.buffers())
        if use_buffers else model.parameters()
    )
    teacher_param = (
        itertools.chain(ema_model.parameters(), ema_model.buffers())
        if use_buffers else ema_model.parameters()
    )

    for ema_param, param in zip(teacher_param, student_param):
        ema_param.detach().copy_(avg_fn(ema_param.detach(), param))


def get_current_consistency_weight(epoch, rampup_length):
    # default parameters
    # --consistency 100.0
    # --consistency-rampup 5
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    res = ramps.sigmoid_rampup(epoch, rampup_length)
    print(f"current consistency weight: {res} at epoch {epoch}", flush=True)
    return res


def main():
    parser = argparse.ArgumentParser("Microscopy image segmentation")
    # Dataset parameters
    parser.add_argument(
        "--labeled_path",
        default="./data/Train_Pre_grad_aug2/",
        type=str,
        help="training data path; subfolders: images, labels",
    )

    parser.add_argument(
        "--unlabeled_path",
        default="./data/Train_Pre_Unlabeled/",
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
        "--model_name", default="coat_daformer_grad_v3", help="select mode: unet, unetr, swinunetr??? swinunetr_dfc_v3"
    )
    parser.add_argument("--num_class", default=4, type=int, help="segmentation classes")
    parser.add_argument(
        "--input_size", default=512, type=int, help="segmentation classes"
    )

    parser.add_argument('--continue_train', required=False, default=False, action="store_true")
    parser.add_argument('--val_spilt_fraction', required=False, type=float, default=0.1)
    # Training parameters
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size per GPU")
    parser.add_argument("--max_epochs", default=800, type=int)
    parser.add_argument("--val_interval", default=2, type=int)
    parser.add_argument("--epoch_tolerance", default=100, type=int)
    parser.add_argument("--initial_lr", type=float, default=6e-5, help="learning rate")
    parser.add_argument("--alpha", type=float, default=0.999, help="ema")
    parser.add_argument("--confidence_threshold", type=float, default=0.50, help="confidence threshold")
    parser.add_argument("--grad_lambda", type=float, default=1, help="grad hyper-parameter")
    parser.add_argument("--vat", required=False, default=False, action="store_true", help="T-VAT enable or not")
    parser.add_argument("--eva_use_student", required=False, default=False, action="store_true", help="eval using student model or not")
    parser.add_argument("--finetune", required=False, default=False, action="store_true", help="finetune will not record the resume checkpoint best val score")
    parser.add_argument("--consistence_w", type=float, required=False, default=2, help="consistence weight")
    parser.add_argument("--rampup_length", type=int, required=False, default=60, help="rampup length")
    parser.add_argument("--lb_consistence", type=int, required=False, default=0, help="labeled image consistence loss: enable->1, disable->0")

    args = parser.parse_args()

    from model_selector import model_factory

    from transforms.utils import ConditionChannelNumberd

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
    model_path = join(args.work_dir, args.model_name + "_mean_teacher")
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

    gt_names = [img_name.split(".")[0] + "_label_flows.tif" for img_name in labeled_img_names]
    img_num = len(labeled_img_names)
    unlabeled_img_num = len(unlabeled_img_names)

    val_frac = args.val_spilt_fraction

    val_split = int(img_num * val_frac)
    indices = np.arange(img_num)
    np.random.shuffle(indices)
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    unlabeled_train_indices = np.arange(unlabeled_img_num)

    train_labeled_files = [
        {"img": join(img_path, labeled_img_names[i]), "label": join(gt_path, gt_names[i])}
        for i in train_indices
    ]

    train_unlabeled_files = [
        {"img": join(unlabeled_img_path, unlabeled_img_names[i])}
        for i in unlabeled_train_indices
    ]

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
                keys=["img"], reader=PILReader(), allow_missing_keys=True
            ),  # image three channels (H, W, 3); label: (H, W)
            LoadImaged(keys=["label"], reader=TiffReader2(channel_dim=0), allow_missing_keys=True),
            Flow2dTransposeFixd(keys=["label"], flow_dim_start=2, flow_dim_end=4, allow_missing_keys=True),

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
            # SpatialPadd(keys=["img", "label"], spatial_size=args.input_size),
            RandAxisFlipd(keys=["img", "label"], prob=0.5, allow_missing_keys=True),
            Flow2dFlipFixd(keys=["label"], flow_dim_start=2, flow_dim_end=4, allow_missing_keys=True),
            RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1], allow_missing_keys=True),
            Flow2dRoatation90Fixd(keys=["label"], flow_dim_start=2, flow_dim_end=4, allow_missing_keys=True),
            # Rand2DElasticd(keys=["img", "label"], spacing=(7, 7), magnitude_range=(-3, 3), mode=[GridSampleMode.BILINEAR, GridSampleMode.NEAREST]),
            # # intensity transform
            # RandRotated(keys=["img", "label"], range_x=(-3.14, 3.14), range_y=(-3.14, 3.14), prob=0.6, mode=[GridSampleMode.BILINEAR, GridSampleMode.NEAREST],
            #             padding_mode=GridSamplePadMode.ZEROS, allow_missing_keys=True),
            # Flow2dRoatateFixd(keys=["label"], flow_dim_start=2, flow_dim_end=4, allow_missing_keys=True),
            RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
            # RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),

            RandZoomd(
                keys=["img", "label"],
                prob=0.5,
                min_zoom=0.2,
                max_zoom=1.8,
                mode=["area", "nearest"],
                padding_mode="constant", allow_missing_keys=True
            ),
            EnsureTyped(keys=["img", "label"], allow_missing_keys=True),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["img"], reader=PILReader()),
            LoadImaged(keys=["label"], reader=TiffReader2(channel_dim=0), allow_missing_keys=True),
            Flow2dTransposeFixd(keys=["label"], flow_dim_start=2, flow_dim_end=4, allow_missing_keys=True),

            EnsureChannelFirstd(
                keys=["img"],
            ),  # image: (3, H, W)
            ConditionChannelNumberd(
                keys=["img"], target_dim=0, channel_num=3, allow_missing_keys=True
            ),

            ScaleIntensityd(keys=["img"], allow_missing_keys=True),
            # AsDiscreted(keys=['label'], to_onehot=3),
            EnsureTyped(keys=["img", "label"], allow_missing_keys=True),
        ]
    )

    at = Compose(
        [
            RandScaleIntensityd(keys=["img"], prob=0.5, factors=0.1, allow_missing_keys=True),
            ColorJitterd(keys=["img"], allow_missing_keys=True),
            RandGaussianNoised(keys=["img"], prob=0.5, mean=0, std=0.1, allow_missing_keys=True),
            RandGaussianSmoothd(keys=["img"], prob=0.5, sigma_x=(1, 2), sigma_y=(1, 2), allow_missing_keys=True),

            EnsureTyped(keys=["img"], allow_missing_keys=True),
            RandCutoutd(keys=["img"], allow_missing_keys=True, prob=0.5)
        ]
    )

    # % define dataset, data loader
    check_ds = DualStreamDataset(labeled_dataset=train_labeled_files, unlabeled_dataset=train_unlabeled_files, weak_aug_transforms=at,
                                 strong_aug_transforms=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=4)
    check_data = monai.utils.misc.first(check_loader)
    print(
        "sanity check:",
        check_data["img"].shape,
        torch.max(check_data["img"]),
        check_data["label"].shape,
        torch.max(check_data["label"]),
    )

    del check_ds
    del check_loader
    del check_data

    # %% create a training data loader
    debug_spilt = len(train_labeled_files)

    mt_train_ds = DualStreamDataset(labeled_dataset=train_labeled_files[:debug_spilt], unlabeled_dataset=train_unlabeled_files[:debug_spilt], weak_aug_transforms=at,
                                    strong_aug_transforms=train_transforms)

    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        mt_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=True
    )

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms, )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)
    dice_metric = DiceMetric(
        include_background=False, reduction="mean", get_not_nans=False
    )
    f1_metric = CellF1Metric(get_not_nans=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create student model
    student_model = model_factory(args.model_name.lower(), device, args, in_channels=3)

    sup_loss_function_seg = monai.losses.DiceCELoss(softmax=True)
    sup_loss_function_grad = sim.MSE()

    # consistency_function_seg = sim.semi_ce_loss
    consistency_function_seg = torch.nn.CrossEntropyLoss()
    consistency_function_grad = sim.ConfidenceMSE()

    # confidence ce
    confidence_threshold = args.confidence_threshold
    # grad model
    grad_lambda = args.grad_lambda

    restart_epoch = 1

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

    # VAT for our grad regression and segmentation
    def model_predictor(new_latent, logits):
        pred_img = student_model.decode(new_latent)
        (pred_mask, pred_grad), (adv_pred_mask, adv_pred_grad) = output2seg_and_grad(pred_img), output2seg_and_grad(logits)
        adv_loss = sim.kl(torch.flatten(pred_mask, start_dim=1), torch.flatten(adv_pred_mask, start_dim=1), reduction="mean")
        adv_loss = adv_loss + grad_lambda * torch.nn.functional.mse_loss(pred_grad, adv_pred_grad)

        return adv_loss

    if args.continue_train:
        load_model_path = join(model_path, 'best_F1_model.pth')
        if not os.path.exists(load_model_path):
            load_model_path = join(model_path, 'best_Dice_model.pth')
        # load model parameters
        checkpoint = torch.load(load_model_path, map_location=torch.device(device))
        student_model.load_state_dict(checkpoint['model_state_dict'])
        if not args.finetune:
            optimizer = torch.optim.AdamW(student_model.parameters(), initial_lr, eps=1e-4)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # torch bug... see github issues
            optimizer.param_groups[0]['capturable'] = True
        restart_epoch = checkpoint['epoch']
        history_loss = checkpoint['loss']
        epoch_loss_values.append(history_loss)
        best_metric_epoch = restart_epoch
        if 'eval_metric' in checkpoint and not args.finetune:
            best_metric = checkpoint['eval_metric']

    ########### mean teacher train process #############
    #  Dual Stream applied at vanilla mean teacher
    #  paper, we mixed unlabeled and labeled images and
    #  the consistency loss will be calculated while the
    #  unlabeled images are sampled.
    ####################################################

    from torch.cuda.amp import GradScaler
    scaler = GradScaler()

    teacher_model_1 = copy.deepcopy(student_model)
    teacher_model_2 = copy.deepcopy(student_model)

    # teacher_model_1.train()
    # teacher_model_2.train()

    if args.vat:
        teacher_model_1.output_latent_enable()
        teacher_model_2.output_latent_enable()

    for epoch in range(restart_epoch, max_epochs):
        student_model.train()
        # teacher_model_1.train()
        # teacher_model_2.train()
        epoch_loss = 0
        train_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        consistence_w = get_current_consistency_weight(epoch, args.rampup_length) * args.consistence_w

        for step, batch_data in train_bar:

            labels = batch_data["label"].to(device)
            lb_img = batch_data["img"].to(device)

            waul_img = batch_data["waul_img"].to(device)
            ul_img = batch_data["ul_img"].to(device)

            cut_mask = batch_data["cutout_mask"].to(device)

            optimizer.zero_grad()

            teacher_model = teacher_model_1
            if epoch % 2 == 0:
                teacher_model = teacher_model_2
            noise = 0
            if args.vat:
                with torch.no_grad():
                    latent_1 = teacher_model_1.encode(ul_img)
                    latent_2 = teacher_model_2.encode(ul_img)
                p_label_1 = teacher_model_1.decode(latent_1)
                p_label_2 = teacher_model_1.decode(latent_2)
                # mask the label from teacher model

                # concat at batch channel
                p_label = cut_mask * (torch.cat([p_label_1, p_label_2]))
                # calculate the noise (of the latent) by forward the decoder of the model and statistic the grad direction of the latent.
                noise = sim.get_vat_noise(model_predictor, (latent_1[-1] + latent_2[-1]) * 0.5, p_label)

            else:
                with torch.no_grad():
                    p_label_1 = teacher_model_1(ul_img).detach()
                    p_label_2 = teacher_model_2(ul_img).detach()

                p_label = cut_mask * (torch.cat([p_label_1, p_label_2]))
                del p_label_1, p_label_2

            imgs = torch.cat([lb_img, waul_img])
            latent = student_model.encode(imgs)

            pred_img = student_model.decode(latent)

            adv_loss = 0

            if args.vat:
                # apply the addition noise
                latent[-1] = latent[-1] + noise

                # minimum the distance of the noised latent and the original latent by the predictor(decoder) outputs.
                # the noise is obtained from teacher model virtual training not student's.
                # todo it is wrong!
                adv_loss = model_predictor(latent[-1].detach(), p_label)
                # adv_pred_img = student_model.decode(latent)
                # (pred_mask, pred_grad), (adv_pred_mask, adv_pred_grad) = output2seg_and_grad(pred_img),  output2seg_and_grad(adv_pred_img)
                # adv_loss = sim.kl(torch.flatten(pred_mask, start_dim=1), torch.flatten(adv_pred_mask, start_dim=1), reduction="mean")
                # adv_loss = adv_loss + grad_lambda * torch.nn.functional.mse_loss(pred_grad, adv_pred_grad)

            pred_lb_img, pred_ul_img = pred_img[:args.batch_size], pred_img[args.batch_size:]

            pred_mask, pred_grad = output2seg_and_grad(pred_lb_img)
            _, labels_onehot, label_grad = label2seg_and_grad(labels)

            pred_ul_mask, pred_ul_grad = output2seg_and_grad(pred_ul_img)
            p_label_mask, p_label_grad = output2seg_and_grad(p_label)

            pred_ul_mask_double = torch.repeat_interleave(pred_ul_mask, 2, dim=0)
            conf_ce_loss = consistency_function_seg(pred_ul_mask_double, torch.softmax(p_label_mask, 1))

            # print(conf_ce_loss, pred_ul_mask_double.shape, p_label_mask.shape)

            consistency_loss = (conf_ce_loss + grad_lambda * consistency_function_grad.loss(p_label_grad, pred_ul_grad, 1)) * consistence_w
            # consistency_loss = conf_ce_loss * consistence_w
            consistency_loss_2 = 0

            # test
            # if args.lb_consistence > 0:
            #     with torch.no_grad():
            #         # add consistence
            #         t_lb_label_1 = teacher_model_1(lb_img).detach()
            #         t_lb_label_2 = teacher_model_2(lb_img).detach()
            #         t_lb_label = 0.5 * (t_lb_label_1 + t_lb_label_2)
            #     t_label_mask, t_label_grad = output2seg_and_grad(t_lb_label)
            #
            #     conf_ce_loss_2 = consistency_function_seg(torch.repeat_interleave(pred_mask, 2, dim=0), t_label_mask)
            #     consistency_loss_2 = (conf_ce_loss_2 + grad_lambda * consistency_function_grad.loss(t_label_grad, pred_grad, 1)) * consistence_w

            sup_loss = sup_loss_function_seg(pred_mask, labels_onehot) + grad_lambda * sup_loss_function_grad.loss(label_grad * 5, pred_grad)

            loss = (consistency_loss + consistency_loss_2) + sup_loss + adv_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_len = len(train_loader)

            global_step = epoch * epoch_len + step

            # update the teacher model at each step
            # update one model
            update_ema_variables(student_model, teacher_model, global_step, alpha=alpha)

            train_bar.set_postfix_str(f"train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), global_step)

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch} average loss: {epoch_loss:.4f}")

        eval_model = student_model if args.eva_use_student else teacher_model_1

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": eval_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss_values,
        }

        if epoch >= 10 and epoch % val_interval == 0:
            eval_model.eval()
            with torch.no_grad():
                val_images_board = None
                val_labels_board = None
                val_outputs_board = None
                val_grad_board = None
                val_label_grad_board = None
                val_instance_label_board = None
                val_pred_instance_label_board = None

                for val_step, val_data in enumerate(val_loader):
                    val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
                    # val_images = stain_model(val_images).to(device)

                    val_instance_label, val_labels_onehot, label_grad = label2seg_and_grad(val_labels)

                    roi_size = (args.input_size, args.input_size)
                    sw_batch_size = args.batch_size

                    val_outputs = sliding_window_inference(
                        val_images, roi_size, sw_batch_size, eval_model, overlap=0.5, mode="gaussian"
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
                                                               cellprob_threshold=0.5, use_gpu=True)

                    # numpy.ndarray -> torch.tensor
                    # H x W -> B x C x H x W
                    instance_label = torch.from_numpy(instance_label.astype(np.float32))[None, None, ...]
                    val_instance_label = val_instance_label.cpu()

                    del pred_grad, label_grad

                    if val_step == epoch % len(val_loader):
                        g_step = max(pred_grad_cpu.shape[1] // 128, 1)
                        flow_ = pred_grad_cpu[:, ::g_step, ::g_step].numpy() / 2

                        fig, _, _ = flow([flow_.transpose(1, 2, 0)], show=False, width=10)
                        val_grad_board = val_grad
                        fig_tensor = fig2data(fig)[:, :, :3].transpose(2, 0, 1)[None, ...]

                        flow_ = label_grad_cpu[:, ::g_step, ::g_step].numpy() * 2
                        fig, _, _ = flow([flow_.transpose(1, 2, 0)], show=False, width=10)
                        # val_grad_board = val_grad
                        label_fig_tensor = fig2data(fig)[:, :, :3].transpose(2, 0, 1)[None, ...]
                        plt.close('all')

                        val_outputs = val_grad * (pred_label_cpu[1].numpy() > 0.5)
                        val_images_board = val_images
                        val_labels_board = val_labels
                        val_outputs_board = val_outputs
                        val_label_grad_board = val_label_grad
                        # val_instance_label_board = val_instance_label
                        # val_pred_instance_label_board = instance_label

                    f1 = f1_metric(y_pred=instance_label, y=val_instance_label)
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
                    checkpoint["eval_metric"] = best_metric
                    torch.save(checkpoint, join(model_path, "best_F1_model.pth"))
                    print("saved new best metric model")

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
