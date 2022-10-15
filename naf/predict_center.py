import os

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

from transforms import flow_gen

from transforms.utils import post_process, post_process_2, post_process_3
import monai.networks

from model_selector import model_factory

join = os.path.join
import argparse
import numpy as np
import torch
from monai.inferers import sliding_window_inference
import time
from skimage import io, segmentation, morphology, measure, exposure
import tifffile as tif


def label2seg_and_center_prob(labels):
    seg_label = labels[:, 4:]
    # seg_label[seg_label > 0] = 1
    labels_onehot = monai.networks.one_hot(
        seg_label, 2
    )

    return labels[:, :1], labels_onehot, labels[:, 1:2]


def output2seg_and_center_prob(outputs):
    return outputs[:, :2], outputs[:, 2:]


def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser('Baseline for Microscopy image segmentation', add_help=False)
    # Dataset parameters
    parser.add_argument('-i', '--input_path', default='./inputs', type=str, help='training data path; subfolders: images, labels')
    parser.add_argument("-o", '--output_path', default='./outputs', type=str, help='output path')
    parser.add_argument('--model_path', default='./work_dir/swinunetrv2_3class', help='path where to save models and segmentation results')
    parser.add_argument('--show_overlay', required=False, default=False, action="store_true", help='save segmentation overlay')
    parser.add_argument('--show_grad', required=False, default=False, action="store_true", help='save segmentation grad')

    # Model parameters
    parser.add_argument('--model_name', default='swinunetrv2', help='select mode: unet, unetr, swinunetr，swinunetrv2')
    parser.add_argument('--num_class', default=3, type=int, help='segmentation classes')
    parser.add_argument('--input_size', default=512, type=int, help='segmentation classes')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    img_names = sorted(os.listdir(join(input_path)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_factory(args.model_name.lower(), device, args, in_channels=3)

    # find best model
    model_path = join(args.model_path, 'best_F1_model.pth')
    if not os.path.exists(model_path):
        model_path = join(args.model_path, 'best_Dice_model.pth')

    print(f"Loading {model_path}...")
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    # %%
    roi_size = (args.input_size, args.input_size)
    sw_batch_size = 2
    model.eval()
    # model = model.half()
    torch.set_grad_enabled(False)
    # print(torch.cuda.memory_summary())
    # torch.cuda.set_per_process_memory_fraction(0.5, 0)

    with torch.no_grad():
        for img_name in img_names:
            # torch.cuda.empty_cache()

            if img_name.endswith('.tif') or img_name.endswith('.tiff'):
                img_data = tif.imread(join(input_path, img_name))
            else:
                img_data = io.imread(join(input_path, img_name))

            # normalize image data
            if len(img_data.shape) == 2:
                img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
            elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
                img_data = img_data[:, :, :3]
            else:
                pass
            pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
            for i in range(3):
                img_channel_i = img_data[:, :, i]
                if len(img_channel_i[np.nonzero(img_channel_i)]) > 0:
                    pre_img_data[:, :, i] = normalize_channel(img_channel_i, lower=1, upper=99)

            t0 = time.time()
            test_npy01 = pre_img_data / np.max(pre_img_data)
            test_tensor = torch.from_numpy(np.expand_dims(test_npy01, 0)).permute(0, 3, 1, 2).type(torch.FloatTensor).to(device)
            test_pred_out = sliding_window_inference(test_tensor, roi_size, sw_batch_size, model, mode="gaussian",
                                                     overlap=0.5)

            pred_label, pred_cent_prob = output2seg_and_center_prob(test_pred_out)

            # pred_label = torch.softmax(pred_label, dim=1)

            pred_label = torch.argmax(pred_label, dim=1)  # (B, C, H, W)
            pred_cent_prob_cpu = pred_cent_prob.detach().cpu()[0].numpy()
            pred_label_cpu = pred_label.detach().cpu()[0].numpy()
            # pred_label_cpu[pred_label_cpu > 0.5] = 1
            # label_grad_cpu = label_grad.detach().cpu()[0]
            pred_cent_prob_cpu[pred_cent_prob_cpu > 0.95] = 1
            pred_cent_prob_cpu[pred_cent_prob_cpu <= 0.95] = 0

            if pred_cent_prob_cpu.shape[2] <= 6000 and pred_cent_prob_cpu.shape[1] <= 6000:
                # test_pred_mask[test_pred_mask > 0] = 1
                # test_pred_mask = post_process_3(morphology.remove_small_objects(morphology.remove_small_holes(pred_label_cpu >= 0.5), 16))

                if pred_cent_prob_cpu.max() == 0:
                    test_pred_mask = measure.label(morphology.remove_small_objects(morphology.remove_small_holes(pred_label_cpu > 0.5), 16))
                else:
                    test_pred_mask = post_process_3(label=pred_label_cpu, cell_prob=0.5, markers=measure.label(pred_cent_prob_cpu[0], connectivity=1))

            else:
                test_pred_mask = measure.label(morphology.remove_small_objects(morphology.remove_small_holes(pred_label_cpu > 0.5), 16))
                # test_pred_mask, _ = flow_gen.compute_masks(pred_cent_prob_cpu,
                #                                            pred_label_cpu,
                #                                            cellprob_threshold=0.5, flow_threshold=1.6, niter=200, use_gpu=False, min_size=16)

                # test_pred_mask = post_process_3(morphology.remove_small_objects(morphology.remove_small_holes(test_pred_mask >= 0.5), 16))
            # test_pred_mask = post_process_2(test_pred_mask, max_size=70 * 70 * 4)
            # test_pred_mask = post_process_3(test_pred_mask)

            # test_pred_npy = test_pred_out[0, 1].cpu().numpy()
            # convert probability map to binary mask and apply morphological postprocessing
            # test_pred_mask = measure.label(morphology.remove_small_objects(morphology.remove_small_holes(test_pred_npy > 0.5), 16))
            # test_pred_mask = post_process(test_pred_mask)
            tif.imwrite(join(output_path, img_name.split('.')[0] + '_label.tiff'), test_pred_mask, compression='zlib')
            t1 = time.time()
            print(f'Prediction finished: {img_name}; img size = {pre_img_data.shape}; costing: {t1 - t0:.2f}s')
            if args.show_grad:
                from transforms.utils import dx_to_circ
                rgb_grad = dx_to_circ(pred_cent_prob_cpu)
                tif.imwrite(join(output_path, 'overlay_' + img_name.split('.')[0] + '_flow.tiff'), rgb_grad, compression='zlib')
            if args.show_overlay:
                boundary = segmentation.find_boundaries(test_pred_mask, connectivity=1, mode='inner')
                boundary = morphology.binary_dilation(boundary, morphology.disk(2))
                img_data[boundary, :] = 255
                io.imsave(join(output_path, 'overlay_' + img_name), img_data, check_contrast=False)


if __name__ == "__main__":
    main()
