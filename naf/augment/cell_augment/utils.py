import glob
import math
import os

import numpy as np
import imageio as io
# taichi's GUI will interpolate position for every particle at the pixels (the position is not always at the middle of the pixel)... we need color the data by ourself.
from monai.transforms import GaussianSmooth, RandGaussianNoise
from numba import jit
from scipy.ndimage import binary_erosion

smooth_transformer = GaussianSmooth(sigma=0.5)
cell_smooth_transformer = GaussianSmooth(sigma=2)
bg_gaussian_noise = RandGaussianNoise(prob=1, std=3, dtype=np.uint8)


def simulation_data2image(path, image_path, size=(256, 256)):
    data = np.load(path)
    datas = data['pos'], data['items']
    data.close()

    res = np.zeros(size)
    res = simulation_data2image_inner(datas, res, size)

    io.imwrite(image_path, res.astype(np.uint8))


@jit(nopython=True)
def clamp(a, min_, max_):
    return min(max(a, min_), max_)


@jit(nopython=True)
def simulation_data2image_inner(data, output, size):
    positions = data[0]
    item = data[1]
    offset = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]

    particle_num = positions.shape[0]
    it_num = item.shape[0]
    assert it_num == particle_num

    for i in range(particle_num):
        xy = positions[i]
        item_idx = item[i]

        h = xy[0] * size[0]
        w = xy[1] * size[1]

        for o in offset:
            x = int(h) + o[0]
            y = int(w) + o[1]
            output[clamp(x, 0, size[0] - 1), clamp(y, 0, size[1] - 1)] = int(item_idx) + 1

    return output


def convert_sim_data2png(data_dir, save_dir, size=(256, 256)):
    files = glob.glob(f"{data_dir}/*.npz")
    for f in files:
        fn = os.path.basename(f)
        save_path = os.path.join(save_dir, fn.replace(".npz", "_label.png"))
        simulation_data2image(f, save_path, size)


import random
import copy


def random_color_inside_cell_label(img, dec=0.5, base_color_dec=.2, base_color=10, ol=128, shadow=128):
    idx_list = list(np.unique(img))

    # remove background
    idx_list.remove(0)

    old_idx_list = copy.deepcopy(idx_list)

    random.shuffle(idx_list)

    output = np.zeros(img.shape + (3,), dtype=np.uint8)

    # add background
    bg = random_background(output.shape)

    output += bg.astype(np.uint8)
    co = 1 / len(idx_list) * math.pi / 2
    R, G, B = 0, 1, 2

    for idx in old_idx_list:
        cell = np.zeros_like(output)
        new_idx = idx_list[idx - 1]
        cell_bool = (img == new_idx)
        r = (base_color + 1 + int(math.sin(idx * co) * 255 * base_color_dec) + int(random.random() * 2)) * dec
        g = (base_color + 1 + int(math.sin(idx * co) * 255 * base_color_dec) + int(random.random() * 2)) * dec
        b = (base_color + 1 + int(math.sin(idx * co) * 255 * base_color_dec) + int(random.random() * 2)) * dec

        cell_bool_erosion_1 = binary_erosion(cell_bool, iterations=1)
        cell_bool_erosion_2 = binary_erosion(cell_bool_erosion_1, iterations=2)

        br = cell_bool_erosion_1 ^ cell_bool_erosion_2
        ob = cell_bool_erosion_1 ^ cell_bool

        cell[..., R][cell_bool_erosion_1] = clamp(r, 0, 255)
        cell[..., G][cell_bool_erosion_1] = clamp(g, 0, 255)
        cell[..., B][cell_bool_erosion_1] = clamp(b, 0, 255)

        cell[..., R][ob] = clamp(r + ol, 0, 255)
        cell[..., G][ob] = clamp(g + ol, 0, 255)
        cell[..., B][ob] = clamp(b + ol, 0, 255)

        cell[..., R][br] = clamp(r - shadow, 0, 255)
        cell[..., G][br] = clamp(g - shadow, 0, 255)
        cell[..., B][br] = clamp(b - shadow, 0, 255)

        smooth_cell = cell_smooth_transformer(cell.transpose(2, 0, 1)).transpose(1, 2, 0).astype(np.uint8)

        output[cell_bool] = 0
        output += smooth_cell

    output = smooth_transformer(output.transpose(2, 0, 1)).transpose(1, 2, 0).astype(np.uint8)

    return output


def random_background(shape, base_color=80):
    bg = np.zeros(shape)

    base_color += random.random() * 10
    bg += base_color
    bg = bg_gaussian_noise(bg)

    return bg


if __name__ == '__main__':
    # convert_sim_data2png("simulation_data", "gen_labels")
    label_dir = "gen_labels"
    image_dir = "gen_img"
    labels = glob.glob("gen_labels/*")
    # for l in labels:
    #     img = io.imread_v2(l)
    #     file_name = os.path.basename(l)
    #     io.imwrite(f"gen_img/{file_name.replace('_label', '')}", random_color_inside_cell_label(img))

    imgs = glob.glob("gen_img/*")

    cell_idx = 1001
    for img, label in zip(imgs, labels):
        idx = "%05d" % cell_idx
        img_name = f"cell_{idx}.png"
        lb_name = f"cell_{idx}_label.png"

        os.rename(img, os.path.join(image_dir, img_name))
        os.rename(label, os.path.join(label_dir, lb_name))

        cell_idx += 1

