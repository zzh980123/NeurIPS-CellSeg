import math
import random
from typing import Union, Tuple

import numpy as np
from monai.utils import ensure_tuple
from numba import jit
import imageio as io


def get_image_template(path: str):
    return io.imread_v2(path)


def get_label_mask(path: str):
    return io.imread_v2(path)


def gray(img):
    if img.ndim == 3 and img.shape[-1] >= 3:
        return img[..., :3] @ np.array([[0.3, 0.6, 0.1]]).reshape((3, 1))

    if img.ndim == 2:
        img = img[..., None]

    return img


def normalize_label(label_mask):
    label_mask[label_mask == 128] = 1.0
    label_mask[label_mask == 255] = 2.0
    return label_mask


def gen_strip_cells(cell_temp_path: str, label_temp_path: str, output_img_size: Union[Tuple, int] = (256, 256), order=1, cell_max_number=100, img_bg_color=0,
                    label_bg_value=0):
    img = get_image_template(path=cell_temp_path)
    lm = get_label_mask(path=label_temp_path)

    lm_normal = gray(normalize_label(lm))
    if img.ndim == 3 and img.shape[-1] > 3:
        img = img[..., :3]

    return _gen_strip_cells(img, lm_normal, output_img_size, order, cell_max_number, img_bg_color, label_bg_value)


def _gen_strip_cells(one_strip_cell_template: np.ndarray, label_mask: np.ndarray, output_img_size: Union[Tuple, int] = (256, 256), order=1, cell_max_number=100, img_bg_color=0,
                     label_bg_value=0):
    cell_shape = one_strip_cell_template.shape
    ndims = one_strip_cell_template.ndim
    assert ndims in [2, 3]

    if ndims == 2:
        one_strip_cell_template = one_strip_cell_template[..., None]

    if isinstance(output_img_size, int):
        output_img_size = (output_img_size,) * ndims
    output_img_size = ensure_tuple(output_img_size)

    return __gen_strip_cells(one_strip_cell_template, label_mask, cell_shape, output_img_size, order, cell_max_number, label_bg_value)


def gen_random_boxes(include_size, box_min_size, box_max_size, number=10, min_angle=0, max_angle=2 * math.pi):

    for _ in range(number):
        random_box_h_scale = random.random()
        random_box_w_scale = random.random()

        inter_h = (1 - random_box_h_scale) * box_min_size[0] + random_box_h_scale * box_max_size[0]
        inter_w = (1 - random_box_w_scale) * box_min_size[1] + random_box_w_scale * box_max_size[1]
        r = math.sqrt(inter_h ** 2 + inter_w ** 2) / 2

    pass


@jit(nopython=True)
def __gen_strip_cells(one_strip_cell_template: np.ndarray, label_mask: np.ndarray, cell_shape, output_img_size: Union[Tuple, int] = (256, 256), order=1, cell_max_number=100,
                      img_bg_color=0,
                      label_bg_value=0):
    C_img = one_strip_cell_template.shape[-1]
    C_label = label_mask.shape[-1]
    output_img = np.empty(output_img_size + (C_img,))
    output_img.fill(img_bg_color)
    output_label = np.empty(output_img_size + (C_label,))
    output_label.fill(label_bg_value)

    if order == 1:
        # filled like:
        # || || || ||
        # || || || ||
        # || || || ||
        # || || || ||

        c_h = cell_shape[0]
        c_w = cell_shape[1]
        h, w = output_img_size[0], output_img_size[1]

        h_n = h // c_h
        w_n = w // c_w

        cnt = 0
        for i in range(h_n):
            for j in range(w_n):
                if cnt > cell_max_number:
                    break
                left_top = [i * c_h, j * c_w]
                rigth_bottom = [left_top[0] + c_h, left_top[1] + c_w]
                output_img[left_top[0]: rigth_bottom[0], left_top[1]: rigth_bottom[1], ] = one_strip_cell_template
                output_label[left_top[0]: rigth_bottom[0], left_top[1]: rigth_bottom[1], ] = label_mask

                cnt += 1

    elif order == 2:

        pass

    return output_img, output_label


if __name__ == '__main__':
    res_img, res_mask = gen_strip_cells('../../../data/ext/cell_00143.png', '../../../data/ext/cell_00143_label.png')

    io.imwrite('../../../data/ext/gen_cell_00143.png', res_img.astype(np.uint8))
    io.imwrite('../../../data/ext/gen_cell_00143_label.png', res_mask.astype(np.uint8))
