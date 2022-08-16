import glob
import math
import os

import numpy as np
import imageio as io
# taichi's GUI will interpolate position for every particle at the pixels (the position is not always at the middle of the pixel)... we need color the data by ourself.
from monai.transforms import GaussianSmooth, RandGaussianNoise
from numba import jit
from scipy.ndimage import binary_erosion

smooth_transformer = GaussianSmooth(sigma=2)
cell_smooth_transformer = GaussianSmooth(sigma=1)
bg_gaussian_noise = RandGaussianNoise(prob=1, std=3, dtype=np.uint8)


def simulation_data2image(path, image_path, size=(256, 256)):
    data = np.load(path)
    datas = data['pos'], data['items']
    data.close()

    res = np.zeros(size)
    res = simulation_data2image_inner(datas, res, size)

    parent_dir = os.path.dirname(image_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
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


def random_color_inside_cell_label(img, dec=1.0, base_color_dec=.1, base_color=10, ol=128, shadow=128, bgc=120, in_1=1, in_2=2):
    idx_list = list(np.unique(img))

    # remove background
    idx_list.remove(0)

    old_idx_list = copy.deepcopy(idx_list)

    random.shuffle(idx_list)

    output = np.zeros(img.shape + (3,), dtype=np.uint8)

    # add background
    bg = random_background(output.shape, bgc)

    output += bg.astype(np.uint8)
    co = 1 / len(idx_list) * math.pi / 4
    R, G, B = 0, 1, 2

    for idx in old_idx_list:
        cell = np.zeros_like(output)

        new_idx = idx_list[idx - 1]
        cell_bool = (img == new_idx)
        r = (base_color + 1 + int(math.sin(idx * co) * 128 * base_color_dec) + int(random.random() * 1)) * dec
        g = (base_color + 1 + int(math.sin(idx * co) * 128 * base_color_dec) + int(random.random() * 1)) * dec
        b = (base_color + 1 + int(math.sin(idx * co) * 128 * base_color_dec) + int(random.random() * 1)) * dec

        cell_bool_erosion_1 = binary_erosion(cell_bool, iterations=in_1)
        cell_bool_erosion_2 = binary_erosion(cell_bool_erosion_1, iterations=in_2)

        br = cell_bool_erosion_1 ^ cell_bool_erosion_2
        ob = cell_bool_erosion_1 ^ cell_bool

        cell_bool_out = np.logical_not(cell_bool)

        cell[..., R][cell_bool] = clamp(r, 0, 255)
        cell[..., G][cell_bool] = clamp(g, 0, 255)
        cell[..., B][cell_bool] = clamp(b, 0, 255)

        cell[..., R][ob] = clamp(r + ol, 0, 255)
        cell[..., G][ob] = clamp(g + ol, 0, 255)
        cell[..., B][ob] = clamp(b + ol, 0, 255)

        cell[..., R][br] = clamp(r - shadow, 0, 255)
        cell[..., G][br] = clamp(g - shadow, 0, 255)
        cell[..., B][br] = clamp(b - shadow, 0, 255)

        cell[..., R][cell_bool_out] = clamp(r + ol, 0, 255)
        cell[..., G][cell_bool_out] = clamp(g + ol, 0, 255)
        cell[..., B][cell_bool_out] = clamp(b + ol, 0, 255)

        smooth_cell = cell_smooth_transformer(cell.transpose(2, 0, 1)).transpose(1, 2, 0).astype(np.uint8)
        smooth_cell[cell_bool_out] = 0
        output[cell_bool] = 0
        output += smooth_cell
        output = np.clip(output, 0, 255)

    output = smooth_transformer(output.transpose(2, 0, 1)).transpose(1, 2, 0).astype(np.uint8)

    return output


def random_background(shape, base_color=80):
    bg = np.zeros(shape)

    base_color += random.random() * 10
    bg += base_color
    bg = bg_gaussian_noise(bg)

    return bg


def check_and_create(path):
    parent_dir = os.path.dirname(path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    return path


def gen_stripe_mid_cells(start_cell_num=1001):
    label_dir = "gen/stripe_mid/gen_labels"
    image_dir = "gen/stripe_mid/gen_imgs"
    simulation_data_dir = "simulation_data_stripe_mid"
    convert_sim_data2png(simulation_data_dir, label_dir)
    return default_image_generate(image_dir, label_dir, start_cell_num=start_cell_num)


def gen_stripe_small_cells(start_cell_num=1034):
    label_dir = "gen/stripe_small_dark/gen_labels"
    image_dir = "gen/stripe_small_dark/gen_imgs"
    simulation_data_dir = "simulation_data_stripe_small"
    convert_sim_data2png(simulation_data_dir, label_dir)

    idx = default_image_generate(image_dir, label_dir, start_cell_num=start_cell_num, base_color=20, ol=20, shadow=40, bg_color=100)
    label_dir = "gen/stripe_small_light/gen_labels"
    image_dir = "gen/stripe_small_light/gen_imgs"
    convert_sim_data2png(simulation_data_dir, label_dir)

    return default_image_generate(image_dir, label_dir, start_cell_num=idx, shadow=120, base_color=220, ol=-255, bg_color=20)


def gen_circle_small_cells(start_cell_num=1114):
    label_dir = "gen/circle_small_dark/gen_labels"
    image_dir = "gen/circle_small_dark/gen_imgs"
    simulation_data_dir = "simulation_data_circle_small"
    convert_sim_data2png(simulation_data_dir, label_dir)

    idx = default_image_generate(image_dir, label_dir, start_cell_num=start_cell_num, base_color=20, ol=50, shadow=40, bg_color=200, in_2=6)
    label_dir = "gen/circle_small_light/gen_labels"
    image_dir = "gen/circle_small_light/gen_imgs"
    convert_sim_data2png(simulation_data_dir, label_dir)

    return default_image_generate(image_dir, label_dir, start_cell_num=idx, shadow=-20, base_color=140, ol=20, bg_color=20, in_2=6)


def default_image_generate(image_dir, label_dir, start_cell_num=1001, base_color=200, ol=220, shadow=100, bg_color=120, in_1=1, in_2=2):
    import tqdm
    labels = glob.glob(f"{label_dir}/*")
    for l in tqdm.tqdm(labels, total=len(labels)):
        img = io.imread_v2(l)
        file_name = os.path.basename(l)
        rand_img = random_color_inside_cell_label(img, dec=0.3, base_color=base_color, ol=ol, shadow=shadow, base_color_dec=0.6, bgc=bg_color, in_1=in_1, in_2=in_2)
        io.imwrite(check_and_create(f"{image_dir}/{file_name.replace('_label', '')}"), rand_img)
        # io.imwrite(f"{image_dir}/{file_name.replace('_label', '_reverse')}", 255 - rand_img)

    imgs = glob.glob(f"{image_dir}/*")

    # rename
    cell_idx = start_cell_num
    for img, label in zip(imgs, labels):
        idx = "%05d" % cell_idx
        img_name = f"cell_{idx}.png"
        lb_name = f"cell_{idx}_label.png"

        os.rename(img, check_and_create(os.path.join(image_dir, img_name)))
        os.rename(label, check_and_create(os.path.join(label_dir, lb_name)))

        cell_idx += 1

    return cell_idx


if __name__ == '__main__':
    # gen_stripe_small_cells()
    gen_circle_small_cells(1114)
