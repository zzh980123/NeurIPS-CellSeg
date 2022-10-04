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

from transforms.utils import dx_to_circ, fig2data

if __name__ == '__main__':
    # label_root = '/media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/Train-Labeled/labels'
    # flow_root = '/media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/Train-Labeled/flows'

    is_3classes = True

    label_root = '../../data/Train_Pre_3class_aug1/labels'
    flow_root = '../../data/Train_Pre_3class_aug1/flows'

    files = glob.glob(label_root + "/*")
    flow_files = [f.replace(label_root, flow_root) for f in files]

    for f, ff in tqdm.tqdm(zip(files, flow_files), total=len(files)):
        # # fixed wrong label from official
        # if "00686" not in f:
        #     continue
        # else:
        #     print("Fixed")
        if is_3classes:
            classes_3 = imageio.imread_v2(f)
            # to two classes
            classes_3[classes_3 > 1] = 0
            instance = measure.label(classes_3, background=0, connectivity=1)
        else:
            instance = imageio.imread_v2(f)

        # only cpu support distance calculation... generate the flows by cpu and the flows will be like:
        # instance label/channel 0,
        # distance map/channel 1,
        # grad map/channel 2,3,
        # semi-segment label/channel 4
        res = flow.labels_to_flows([instance], [ff], use_gpu=False, device="cuda:3")

        # # visual result
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
