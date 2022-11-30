#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

join = os.path.join
import argparse

from skimage import io, segmentation, morphology, exposure
import numpy as np
import tifffile as tif
from tqdm import tqdm
from PIL import ImageFile


def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)



def main():
    parser = argparse.ArgumentParser('Preprocessing for microscopy image segmentation', add_help=False)
    parser.add_argument('-i', '--input_path', default='./data/Train_Unlabeled', type=str, help='training data path; subfolders: images, labels')
    parser.add_argument("-o", '--output_path', default='./data/Train_Pre_Unlabeled', type=str, help='preprocessing data path')
    args = parser.parse_args()

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    source_path = args.input_path
    target_path = args.output_path

    img_path = join(source_path)

    img_names = sorted(os.listdir(img_path))

    pre_img_path = join(target_path)
    os.makedirs(pre_img_path, exist_ok=True)
    bar = tqdm(img_names)
    for img_name in bar:

        save_img = join(target_path, img_name.split('.')[0] + '.png')

        if os.path.exists(save_img):
            continue
        bar.set_postfix_str(img_name)
        if img_name.endswith('.tif') or img_name.endswith('.tiff'):
            img_data = tif.imread(join(img_path, img_name))
        else:
            img_data = io.imread(join(img_path, img_name))

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

        io.imsave(join(target_path, img_name.split('.')[0] + '.png'), pre_img_data.astype(np.uint8), check_contrast=False)


if __name__ == "__main__":
    main()
