import sys,os

import glob
import time

from skimage import measure

import transforms

import transforms.flow_gen as flow
import tifffile
import numpy as np

import matplotlib.pyplot as plot
import tqdm
import imageio

from skimage import io, segmentation, morphology, exposure

from naf.transforms.utils import dx_to_circ, fig2data

def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')

    else:
        img_norm = img
    return img_norm.astype(np.uint8)

def create_interior_map(inst_map):
    """
    Parameters
    ----------
    inst_map : (H,W), np.int16
        DESCRIPTION.

    Returns
    -------
    interior : (H,W), np.uint8 
        three-class map, values: 0,1,2
        0: background
        1: interior
        2: boundary
    """
    # create interior-edge map
    boundary = segmentation.find_boundaries(inst_map, mode='inner')
    boundary = morphology.binary_dilation(boundary, morphology.disk(1))

    interior_temp = np.logical_and(~boundary, inst_map > 0)
    # interior_temp[boundary] = 0
    interior_temp = morphology.remove_small_objects(interior_temp, min_size=16)
    interior = np.zeros_like(inst_map, dtype=np.uint8)
    interior[interior_temp] = 1
    interior[boundary] = 2
    return interior

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Preprocessing for microscopy image segmentation', add_help=False)
    parser.add_argument('-i', '--input_path', default='./data/Train_Labeled/labels', type=str, help='training labels path')
    parser.add_argument("-o", '--output_path', default='./data/Train_Pre_flows/labels', type=str, help='preprocessing data path')
    parser.add_argument("--instance_convert", action="store_true", help="If the input_path is the 3class-labels, try to convert them to instance labels")

    args = parser.parse_args()

    is_3classes = args.instance_convert
    label_root = args.input_path
    flow_root = args.output_path

    files = glob.glob(label_root + "/*")
    
    flow_files = [f.replace(label_root, flow_root) for f in files]

    for f, ff in tqdm.tqdm(zip(files, flow_files), total=len(files)):
        if is_3classes:
            classes_3 = imageio.imread_v2(f)
            # to two classes
            classes_3[classes_3 > 1] = 0
            instance = measure.label(classes_3, background=0, connectivity=1)
        else:
            instance = imageio.imread_v2(f)

        # Only when the device is 'cpu' support distance calculation... generate the flows by cpu, and the flows will be like:
        # instance label/channel 0,
        # distance map/channel 1,
        # grad map/channel 2,3,
        # semi-segment label/channel 4
        # example code:
        # res = flow.labels_to_flows([instance], [ff], use_gpu=False, device="cpu")

        res = flow.labels_to_flows([instance], [ff], use_gpu=False, device="cuda:0")
        instance_label = create_interior_map(res[0][..., 0])
        res[0] = instance_label
        
        # If you want to see the resluts pealse remove the "#" before the codes.
        # # visual results
        # for s in res:
        #     grad_yx = s[2:4, :, :]
        #     im = dx_to_circ(grad_yx)
        #     # print(im.shape)
        #     sss = time.time_ns()
        #     # grad_yx[:, :1] *= -1
        #     # grad_yx[:, 1:] *= -1
        #     # grad_y = np.copy(grad_yx[:, :1])
        #     # grad_x = np.copy(grad_yx[:, 1:])
        #     #
        #     # grad_yx = np.concatenate([grad_x, grad_y], axis=1)
        #
        #     import math
        #
        #     flow_ = grad_yx.transpose(1, 2, 0)
        #
        #     print(np.allclose(flow_.transpose(2, 0, 1), grad_yx))
        #
        #     theta = math.pi / 2 * 3
        #     rotate_matrix = np.array([
        #         [math.cos(theta), -math.sin(theta)],
        #         [math.sin(theta), math.cos(theta)],
        #     ])
        #
        #     # fig, _, _ = transformers.utils.flow([flow_], show=True, width=15)
        #     # val_grad_board = val_grad
        #     # label_fig_tensor1 = fig2data(fig)[:, :, :3]
        #     H, W, C = flow_.shape
        #     # flow_90 = np.rot90(flow_, k=3)
        #     # flow_90 = rotate_matrix @ flow_90.transpose(2, 0, 1).reshape(C, H * W)
        #     # flow_90 = flow_90.reshape(C, W, H).transpose(1, 2, 0)
        #
        #     # flow_t = flow_.transpose(1, 0, 2)
        #     # flow_t[:, :, 0:1] *= -1
        #     # flow_t = rotate_matrix @ flow_t.transpose(2, 0, 1).reshape(C, H * W)
        #     # flow_t = flow_t.reshape(C, W, H).transpose(1, 2, 0)
        #
        #     # flow_f = np.fliplr(flow_)
        #     # flow_f[:, :, :1] *= -1
        #     # flow_f = flow_f @ rotate_matrix
        #     # flow_f = np.flip(flow_, 0)
        #     # flow_f[:, :, 1:] *= -1
        #     # H, W, C = flow_f.shape
        #     # flow_f = rotate_matrix @ flow_f.transpose(2, 0, 1).reshape(C, H * W)
        #     # flow_f = flow_f.reshape(C, H, W).transpose(1, 2, 0)
        #
        #     # res = np.concatenate([flow_90[:,:, :1], flow_90[:,:, 1:0]], axis=2)
        #
        #     fig, _, _ = transforms.utils.flow([flow_[::32, ::32]], show=False, width=15)
        #
        #     # val_grad_board = val_grad
        #     # label_fig_tensor2 = fig2data(fig)[:, :, :3]
        #
        #     # recov, _ = flow.compute_masks(flow_t.transpose(2, 0, 1) * 2, cellprob=s[1].transpose(), cellprob_threshold=0.5, use_gpu=True, device="cuda:3", flow_threshold=0.4)
        #     recov, _ = flow.compute_masks(flow_.transpose(2, 0, 1) * 3, cellprob=s[1], cellprob_threshold=0.5, use_gpu=True, device="cuda:3", flow_threshold=0.4)
        #
        #     print(recov.shape)
        #     print((time.time_ns() - sss) / 1000000)
        #     # plot.imshow(im)
        #     # plot.show()
        #     print(s.shape)
        #     plot.imshow(np.column_stack([s[:1].transpose(1, 2, 0), s[1:2].transpose(1, 2, 0)]))
        #     plot.show()
        #
        #     # plot.imshow(label_fig_tensor1)
        #     # plot.show()
        #     # plot.imshow(label_fig_tensor2)
        #     # plot.show()
