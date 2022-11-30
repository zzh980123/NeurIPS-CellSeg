import os

import matplotlib
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from glob import glob

from PIL import Image

from augment.stain_augment.StainNet.models import StainNet

import torch
import numpy as np
import matplotlib.pyplot as plt
# matplotlib.use("GTK3Agg")
import staintools
import imageio as io
import cv2
import os

def norm(image):
    image = np.array(image).astype(np.float32)
    image = image.transpose((2, 0, 1))
    image = ((image / 255) - 0.5) / 0.5
    image = image[np.newaxis, ...]
    image = torch.from_numpy(image)
    return image


def un_norm(image):
    image = image.cpu().detach().numpy()[0]
    image = np.clip((image * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8).transpose((1, 2, 0))
    return image


def create_stain_model(mode=0):
    assert mode in [0, 1, 2]
    checkpoint1 = "./StainNet/checkpoints/aligned_cytopathology_dataset/StainNet-3x0_best_psnr_layer3_ch32.pth"
    checkpoint2 = "./StainNet/checkpoints/aligned_histopathology_dataset/StainNet-Public_layer3_ch32.pth"
    checkpoint3 = "./StainNet/checkpoints/camelyon16_dataset/StainNet-Public-centerUni_layer3_ch32.pth"

    checkpoints = [
        checkpoint1, checkpoint2, checkpoint3
    ]

    model_Net = StainNet()
    model_Net.load_state_dict(torch.load(checkpoints[mode]))
    return model_Net


if __name__ == '__main__':
    # staintools.ReinhardColorNormalizer()
    print(os.getcwd())
    normalizer = staintools.ReinhardColorNormalizer()
    model_Net = StainNet().cuda()
    checkpoint1 = "./naf/augment/stain_augment/StainNet/checkpoints/aligned_cytopathology_dataset/StainNet-3x0_best_psnr_layer3_ch32.pth"
    checkpoint2 = "./naf/augment/stain_augment/StainNet/checkpoints/aligned_histopathology_dataset/StainNet-Public_layer3_ch32.pth"
    checkpoint3 = "./naf/augment/stain_augment/StainNet/checkpoints/camelyon16_dataset/StainNet-Public-centerUni_layer3_ch32.pth"
    model_Net.load_state_dict(torch.load(checkpoint2))

    test_img1 = '/media/kevin/870A38D039F26F71/Datasets/hubmap-organ-segmentation/test_images/10078.tiff'
    test_img3 = '/media/kevin/870A38D039F26F71/Datasets/hubmap-organ-segmentation/HuBMAP_train/images/12466.png'
    test_img2 = '/media/kevin/870A38D039F26F71/PycharmProjects/CellSeg/NeurIPS-CellSeg/data/Train_Pre_cell_size/images/cell_00169.png'

    img_paths = glob("/media/kevin/870A38D039F26F71/Datasets/hubmap-organ-segmentation/HuBMAP_train/images/*.png")
    # img_paths = glob("../../../data/Train_Pre_Unlabeled/*")
    # img_paths.sort()
    # img_paths = [test_img1]
    import time
    for img_path in img_paths:
        img_source = io.imread_v2(img_path)
        img_target = io.imread_v2(test_img1)
        # st = time.time_ns()
        normalizer.fit(img_target)
        res1 = normalizer.transform(img_source)
        # print("ReinhardColorNormalizer:", (time.time_ns() - st) / 1000000)
        st = time.time_ns()
        res2 = model_Net(norm(img_source).cuda())
        res2 = un_norm(res2)
        print("StainNet", (time.time_ns() - st) / 1000000)
        print(os.path.basename(img_path))
        img = np.column_stack([img_source, res2, res1])
        # plt.imshow(img)
        # plt.show()
        # io.imwrite(os.path.join("/media/kevin/870A38D039F26F71/Datasets/hubmap-organ-segmentation/HuBMAP_train/transformed", os.path.basename(img_path)), res2)

