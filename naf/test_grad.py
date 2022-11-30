import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from torchvision.transforms import ColorJitter

from transforms import utils
from transforms import flow_gen

from transforms.utils import post_process_3
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
import memory_profiler as pf
import mem_track

def label2seg_and_grad(labels):
    seg_label = labels[:, 1:2]
    seg_label[seg_label > 0] = 1
    labels_onehot = monai.networks.one_hot(
        seg_label, 2
    )

    return labels[:, :1], labels_onehot, labels[:, 2:]


def output2seg_and_grad(outputs):
    return outputs[:, :2], outputs[:, 2:]


def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)


def tta_enhance(input_img):
    # color jitter
    colorconvert = ColorJitter(brightness=0.2, hue=0.2)

    color_jitter1 = colorconvert(input_img)
    color_jitter2 = colorconvert(input_img)
    color_jitter3 = colorconvert(input_img)
    # rotate_180_img = torch.rot90(input_img, k=2, dims=(2, 3))
    # rotate_270_img = torch.rot90(input_img, k=3, dims=(2, 3))
    batch_imgs = torch.cat([input_img, color_jitter1, color_jitter2, color_jitter3], dim=0)

    return batch_imgs


def tta_enhance_invert(out_img):
    # Rotate the image at batch=2
    # out_img[1:2] = rotate_result(out_img=out_img[1:2], rank_k=4 - 1)
    # out_img[2:3] = rotate_result(out_img=out_img[2:3], rank_k=4 - 2)
    # out_img[3:4] = rotate_result(out_img=out_img[3:4], rank_k=4 - 3)

    return torch.mean(out_img, dim=0, keepdim=True)


################################# rotate enhance ############################################
#
# def tta_enhance(input_img):
#     # rotate
#     rotate_90_img = torch.rot90(input_img, k=1, dims=(2, 3))
#     rotate_180_img = torch.rot90(input_img, k=2, dims=(2, 3))
#     rotate_270_img = torch.rot90(input_img, k=3, dims=(2, 3))
#     batch_imgs = torch.cat([input_img, rotate_90_img, rotate_180_img, rotate_270_img], dim=0)
#
#     return batch_imgs
#
#
# def rotate_result(out_img, rank_k=1):
#     out_img = torch.rot90(out_img, rank_k, dims=(2, 3))
#     seg_label, grad_label = output2seg_and_grad(out_img)
#     # Fix grad label
#     grad_label = utils.rotate_flow(grad_label, rank_k=rank_k)
#
#     out_img[:, 2:] = grad_label
#     return out_img
#
#
# def tta_enhance_invert(out_img):
#     # Rotate the image at batch=2
#     out_img[1:2] = rotate_result(out_img=out_img[1:2], rank_k=4 - 1)
#     out_img[2:3] = rotate_result(out_img=out_img[2:3], rank_k=4 - 2)
#     out_img[3:4] = rotate_result(out_img=out_img[3:4], rank_k=4 - 3)
#
#     return torch.mean(out_img, dim=0, keepdim=True)
    
def main():
    parser = argparse.ArgumentParser('Baseline for Microscopy image segmentation', add_help=False)
    # Dataset parameters
    parser.add_argument('-i', '--input_path', default='./inputs', type=str, help='')
    parser.add_argument('-l', '--label_path', default='./labels', type=str, help='')
    parser.add_argument("-o", '--output_path', default='./outputs', type=str, help='output path')
    parser.add_argument('--model_path', default='./work_dir/swinunetrv2_3class', help='path where to save models and segmentation results')
    parser.add_argument('--show_overlay', required=False, default=False, action="store_true", help='save segmentation overlay')
    parser.add_argument('--show_grad', required=False, default=False, action="store_true", help='save segmentation grad')

    # Model parameters
    parser.add_argument('--model_name', default='swinunetrv2', help='select mode: unet, unetr, swinunetr，swinunetrv2')
    parser.add_argument('--num_class', default=4, type=int, help='segmentation classes')
    parser.add_argument('--input_size', default=512, type=int, help='segmentation classes')
    # some enhance
    parser.add_argument('--watershed', required=False, default=False, action="store_true")
    parser.add_argument('--tta', required=False, default=False, action="store_true")
    parser.add_argument('--use_mask_only', required=False, default=False, action="store_true")

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    gt_path = args.label_path
    os.makedirs(output_path, exist_ok=True)
    img_names = sorted(os.listdir(join(input_path)))
    label_names = [os.path.basename(i).replace(".png", "_label_flows.tif") for i in img_names]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_factory(args.model_name.lower(), device, args, in_channels=3)

    # find best model
    model_path = join(args.model_path, 'best_F1_model.pth')
    if not os.path.exists(model_path):
        model_path = join(args.model_path, 'best_Dice_model.pth')

    print(f"Loading {model_path}...")
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])

    # print(checkpoint['eval_metric'])
    # %%
    roi_size = (args.input_size, args.input_size)
    sw_batch_size = 1
    model.eval()
    # model = model.half()
    torch.set_grad_enabled(False)
    # print(torch.cuda.memory_summary())
    # torch.cuda.set_per_process_memory_fraction(0.5, 0)

    # check the parameters and FLOPs
    # from thop import profile
    # fake_data = torch.randn((1, 3,) + roi_size, device=device)
    # flops, parameters = profile(model, inputs=(fake_data,))
    # print(f"FLOPs: {flops / 1e9} G, parameters: {parameters / 1e6} M")

    mt = mem_track.MemTracker()
    total_time = 0
    with torch.no_grad():
        for img_name, label_name in zip(img_names, label_names):
            # torch.cuda.empty_cache()

            if img_name.endswith('.tif') or img_name.endswith('.tiff'):
                img_data = tif.imread(join(input_path, img_name))
            else:
                img_data = io.imread(join(input_path, img_name))

            if label_name.endswith('.tif') or label_name.endswith('.tiff'):
                label_data = tif.imread(join(gt_path, label_name))
            else:
                label_data = io.imread(join(gt_path, label_name))

            label_mask = label_data[4, ...].T

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
            mt.track()
            test_npy01 = pre_img_data / np.max(pre_img_data)
            test_tensor = torch.from_numpy(np.expand_dims(test_npy01, 0)).permute(0, 3, 1, 2).type(torch.FloatTensor).to(device)

            small_img_flag = test_tensor.shape[2] <= roi_size[0] and test_tensor.shape[3] <= roi_size[1]

            # TTA
            if small_img_flag and args.tta:
                print("TTA")
                test_tensor = tta_enhance(test_tensor)
            
            test_pred_out = sliding_window_inference(test_tensor, roi_size, sw_batch_size, model, mode="gaussian",
                                                     overlap=0.5, device='cpu')
            if small_img_flag and args.tta:
                test_pred_out = tta_enhance_invert(test_pred_out)
                
            pred_label, pred_grad = output2seg_and_grad(test_pred_out)

            pred_label = pred_label[:, 1]  # (B, C,
            pred_grad_cpu = pred_grad.detach().cpu()[0].numpy()
            pred_label_cpu = pred_label.detach().cpu()[0].numpy()

            prob_threshold = 0.475

            if args.tta and small_img_flag:
                prob_threshold = 0.475

            # pred_label_cpu[pred_label_cpu > 0.5] = 1
            # label_grad_cpu = label_grad.detach().cpu()[0]
            pred_grad_cpu = pred_grad_cpu / (np.sqrt(pred_grad_cpu[:1] ** 2 + pred_grad_cpu[1:] ** 2) + 1e-5) * 5

            if pred_grad_cpu.shape[2] <= 6000 and pred_grad_cpu.shape[1] <= 6000 and not args.use_mask_only:
                # test_pred_mask[test_pred_mask > 0] = 1
                # test_pred_mask = post_process_3(morphology.remove_small_objects(morphology.remove_small_holes(pred_label_cpu >= 0.5), 16))
                if args.watershed:
                    # 77.40
                    markers, _ = flow_gen.compute_masks(pred_grad_cpu,
                                                        pred_label_cpu,
                                                        cellprob_threshold=prob_threshold, flow_threshold=1e6, niter=3, use_gpu=True, min_size=2)

                    # markers, _ = flow_gen.compute_masks(pred_grad_cpu,
                    #                                            pred_label_cpu,
                    #                                            cellprob_threshold=0.5, flow_threshold=1e9, niter=3, use_gpu=True, min_size=1)

                    #
                    # test_pred_mask, _ = flow_gen.compute_masks(pred_grad_cpu,
                    #                                            pred_label_cpu,
                    #                                            cellprob_threshold=0.5, use_gpu=True)

                    if markers.max() == 0:
                        markers = measure.label(morphology.remove_small_objects(morphology.remove_small_holes(pred_label_cpu > 0.5), 16), connectivity=1)
                    test_pred_mask = post_process_3(label=pred_label_cpu, cell_prob=prob_threshold, markers=markers)
                else:
                    test_pred_mask, _ = flow_gen.compute_masks(pred_grad_cpu,
                                           pred_label_cpu,
                                           cellprob_threshold=prob_threshold, niter=400, flow_threshold=1.6, use_gpu=True, min_size=16)
                    if test_pred_mask.max() == 0:
                        test_pred_mask = measure.label(morphology.remove_small_objects(morphology.remove_small_holes(pred_label_cpu > 0.5), 16), connectivity=1)
            else:
                test_pred_mask = measure.label(morphology.remove_small_objects(morphology.remove_small_holes(pred_label_cpu > 0.5), 16), connectivity=1)
                # test_pred_mask, _ = flow_gen.compute_masks(pred_grad_cpu,
                #                                            pred_label_cpu,
                #                                            cellprob_threshold=0.4, flow_threshold=2, niter=200, use_gpu=False, min_size=16)

            # print(f"{img_name} count: {test_pred_mask.max().item()}.")
            
            tif.imwrite(join(output_path, img_name.split('.')[0] + '_label.tiff'), test_pred_mask, compression='zlib')
            t1 = time.time()
            mt.track()
            print(f'Prediction finished: {img_name}; img size = {pre_img_data.shape}; costing: {t1 - t0:.2f}s')
            total_time += t1 - t0
            if args.show_grad:
                from transforms.utils import dx_to_circ
                rgb_grad = dx_to_circ(pred_grad_cpu)
                tif.imwrite(join(output_path, 'overlay_' + img_name.split('.')[0] + '_flow.tiff'), rgb_grad, compression='zlib')
            if args.show_overlay:
                boundary = segmentation.find_boundaries(test_pred_mask, connectivity=1, mode='inner')
                boundary = morphology.binary_dilation(boundary, morphology.disk(2))

                boundary_gt = segmentation.find_boundaries(label_mask, connectivity=1, mode='inner')
                boundary_gt = morphology.binary_dilation(boundary_gt, morphology.disk(4))

                img_data[boundary_gt, :] = [255, 0, 0]
                img_data[boundary, :] = [255, 255, 0]

                io.imsave(join(output_path, 'overlay_' + img_name), img_data, check_contrast=False)
            mt.clear_cache()
        print("total time: ", total_time)

if __name__ == "__main__":
    main()