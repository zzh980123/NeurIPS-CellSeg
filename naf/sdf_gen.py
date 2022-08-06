import argparse

import os
import pathlib
from os.path import join

import imageio
import taichi as ti
from monai.transforms import Resize

from skimage import exposure
from tqdm import tqdm

# Obtained from
# https://github.com/hooyuser/2D_SDF_from_mask_GPU/blob/master/JFA_SDF.py
# SDF Generator to convert the GroundTruth to SDF
# 127.5 is the outline, and 0 ~ 127.5 is inside, otherwise is outside.


ti.init(arch=ti.cpu, device_memory_GB=48.0, kernel_profiler=False, debug=False, print_ir=False)

MAX_DIST = 2147483647
null = ti.Vector([-1, -1, MAX_DIST])
vec3 = lambda scalar: ti.Vector([scalar, scalar, scalar])
eps = 1e-5


def resize2d(img, new_size=None):
    resize_flag = new_size is not None

    if img.ndim == 2:
        img = img[None]
    if img.ndim == 3 and img.shape[-1] <= 3:
        img = img.transpose(2, 0, 1)

    H, W, C = img.shape
    size_ = max(H, W)
    if new_size is None:
        new_size = (size_, size_)
    if len(new_size) == 3:
        new_size = new_size[:-1]

    if H != W or resize_flag:
        resize_img = Resize(spatial_size=new_size, mode="nearest")(img).transpose(1, 2, 0)
        return resize_img

    return img


@ti.data_oriented
class SDF2D:
    def __init__(self, filename):
        self.filename = filename
        self.num = 0  # index of bit_pic
        old_img = imageio.imread_v2(filename)
        self.old_shape = old_img.shape
        self.im = resize2d(old_img)
        self.width, self.height = self.im.shape[1], self.im.shape[0]
        self.pic = ti.Vector.field(3, dtype=ti.i32, shape=(self.width, self.height))
        self.bit_pic_white = ti.Vector.field(3, dtype=ti.i32, shape=(2, self.width, self.height))
        self.bit_pic_black = ti.Vector.field(3, dtype=ti.i32, shape=(2, self.width, self.height))
        self.output_pic = ti.Vector.field(3, dtype=ti.i32, shape=(self.width, self.height))
        self.output_linear = ti.Vector.field(3, dtype=ti.f32, shape=(self.width, self.height))
        self.max_reduction = ti.field(dtype=ti.i32, shape=self.width * self.height)

    def __init__(self, image_, filename):
        self.filename = filename
        self.num = 0  # index of bit_pic
        old_img = image_
        self.old_shape = old_img.shape
        self.im = resize2d(old_img)
        self.width, self.height = self.im.shape[1], self.im.shape[0]
        self.pic = ti.Vector.field(3, dtype=ti.i32, shape=(self.width, self.height))
        self.bit_pic_white = ti.Vector.field(3, dtype=ti.i32, shape=(2, self.width, self.height))
        self.bit_pic_black = ti.Vector.field(3, dtype=ti.i32, shape=(2, self.width, self.height))
        self.output_pic = ti.Vector.field(3, dtype=ti.i32, shape=(self.width, self.height))
        self.output_linear = ti.Vector.field(3, dtype=ti.f32, shape=(self.width, self.height))
        self.max_reduction = ti.field(dtype=ti.i32, shape=self.width * self.height)

    def reset(self, filename):
        self.filename = filename
        self.num = 0  # index of bit_pic

        self.im = imageio.imread_v2(filename)
        self.width, self.height = self.im.shape[1], self.im.shape[0]

    def output_filename(self, ins):
        path = pathlib.Path(self.filename)
        out_dir = path.parent / 'output'
        if not (out_dir.exists() and out_dir.is_dir()):
            out_dir.mkdir()
        return str(out_dir / (path.stem + ins + path.suffix))

    @ti.kernel
    def pre_process(self, bit_pic: ti.template(), keep_white: ti.i32):  # keep_white, 1 == True, -1 == False
        for i, j in self.pic:
            if (self.pic[i, j][0] - 127) * keep_white > 0:
                bit_pic[0, i, j] = ti.Vector([i, j, 0])
                bit_pic[1, i, j] = ti.Vector([i, j, 0])
            else:
                bit_pic[0, i, j] = null
                bit_pic[1, i, j] = null

    @ti.func
    def cal_dist_sqr(self, p1_x, p1_y, p2_x, p2_y):
        return (p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2

    @ti.kernel
    def jump_flooding(self, bit_pic: ti.template(), stride: ti.i32, n: ti.i32):
        # print('n =', n, '\n')
        for i, j in ti.ndrange(self.width, self.height):
            for di, dj in ti.ndrange((-1, 2), (-1, 2)):
                i_off = i + stride * di
                j_off = j + stride * dj
                if 0 <= i_off < self.width and 0 <= j_off < self.height:
                    dist_sqr = self.cal_dist_sqr(i, j, bit_pic[n, i_off, j_off][0],
                                                 bit_pic[n, i_off, j_off][1])
                    if not bit_pic[n, i_off, j_off][0] < 0 and dist_sqr < bit_pic[1 - n, i, j][2]:
                        bit_pic[1 - n, i, j][0] = bit_pic[n, i_off, j_off][0]
                        bit_pic[1 - n, i, j][1] = bit_pic[n, i_off, j_off][1]
                        bit_pic[1 - n, i, j][2] = dist_sqr

    @ti.kernel
    def copy(self, bit_pic: ti.template()):
        for i, j in ti.ndrange(self.width, self.height):
            self.max_reduction[i * self.width + j] = bit_pic[self.num, i, j][2]

    @ti.kernel
    def max_reduction_kernel(self, r_stride: ti.i32):
        for i in range(r_stride):
            self.max_reduction[i] = max(self.max_reduction[i], self.max_reduction[i + r_stride])

    @ti.kernel
    def post_process_udf(self, bit_pic: ti.template(), n: ti.i32, coff: ti.f32, offset: ti.f32):
        for i, j in self.output_pic:
            self.output_pic[i, j] = vec3(ti.cast(ti.sqrt(bit_pic[n, i, j][2]) * coff + offset, ti.u32))

    @ti.kernel
    def post_process_sdf(self, bit_pic_w: ti.template(), bit_pic_b: ti.template(), n: ti.i32):
        for i, j in self.output_pic:
            self.output_pic[i, j] = vec3((ti.sqrt(bit_pic_w[n, i, j][2]) - ti.sqrt(bit_pic_b[n, i, j][2])))

    @ti.kernel
    def post_process_sdf_linear_1channel(self, bit_pic_w: ti.template(), bit_pic_b: ti.template(), n: ti.i32):
        for i, j in self.output_pic:
            self.output_linear[i, j][0] = ti.sqrt(bit_pic_w[n, i, j][2]) - ti.sqrt(bit_pic_b[n, i, j][2])

    def gen_udf(self, dist_buffer, keep_white=True):

        keep_white_para = 1 if keep_white else -1
        self.pre_process(dist_buffer, keep_white_para)
        self.num = 0
        stride = self.width >> 1
        while stride > 0:
            self.jump_flooding(dist_buffer, stride, self.num)
            stride >>= 1
            self.num = 1 - self.num

        self.jump_flooding(dist_buffer, 2, self.num)
        self.num = 1 - self.num

        self.jump_flooding(dist_buffer, 1, self.num)
        self.num = 1 - self.num

    def find_max(self, dist_buffer):
        self.copy(dist_buffer)

        r_stride = self.width * self.height >> 1
        while r_stride > 0:
            self.max_reduction_kernel(r_stride)
            r_stride >>= 1

        return self.max_reduction[0]

    def mask2udf(self, normalized=(0, 1), to_rgb=True, output=True, save_path=None):  # unsigned distance
        self.pic.from_numpy(self.im)
        self.gen_udf(self.bit_pic_white)

        max_dist = ti.sqrt(self.find_max(self.bit_pic_white))

        if to_rgb:  # scale sdf proportionally to [0, 1]
            coefficient = 255.0 / max_dist
            offset = 0.0
        else:
            coefficient = (normalized[1] - normalized[0]) / max_dist
            offset = normalized[0]

        self.post_process_udf(self.bit_pic_white, self.num, coefficient, offset)
        if output:
            if to_rgb:
                imageio.imwrite(save_path, resize2d(self.output_pic.to_numpy(), new_size=self.old_shape)[..., 0])

    def gen_udf_w_h(self):
        self.pic.from_numpy(self.im)
        self.gen_udf(self.bit_pic_white, keep_white=True)
        self.gen_udf(self.bit_pic_black, keep_white=False)

    def mask2sdf(self, to_rgb=True, output=True, save_path=None):
        self.gen_udf_w_h()

        if to_rgb:  # grey value == 0.5 means sdf == 0, scale sdf proportionally
            self.post_process_sdf(self.bit_pic_white, self.bit_pic_black, self.num)
            if output:
                pic = self.output_pic.to_numpy()
                coefficient = 127.5 / max(pic.max(), -pic.min())
                pic = pic * coefficient + 127.5
                imageio.imwrite(save_path, resize2d(pic.astype(np.uint8), new_size=self.old_shape)[..., 0])
        else:  # no normalization
            if output:
                pass
            else:
                self.post_process_sdf_linear_1channel(self.bit_pic_white, self.bit_pic_black, self.num)


@ti.data_oriented
class MultiSDF2D:
    def __init__(self, file_name, file_num, sample_num=256, thresholds=None):
        self.file_name = file_name
        self.file_path = pathlib.Path(file_name)
        self.thresholds_tuple = thresholds
        self.file_num = file_num
        self.sample_num = sample_num
        self.name_base = self.file_path.stem[:-2]
        self.file_name_list = self.gen_file_list()
        self.sdf_2d = SDF2D(self.file_name_list[0])
        self.width, self.height = self.sdf_2d.width, self.sdf_2d.height
        self.sdf_buffer = ti.field(dtype=ti.f32, shape=(self.width, self.height, file_num))
        self.output_pic = ti.Vector.field(3, dtype=ti.i32, shape=(self.width, self.height))
        self.thresholds = ti.field(dtype=ti.i32, shape=file_num)

    def calc_thresholds(self):
        if self.thresholds_tuple:
            diff = self.thresholds_tuple[-1] - self.thresholds_tuple[0]
            for i in range(self.file_num):
                self.thresholds[i] = int(self.thresholds_tuple[i] / diff * self.sample_num)
                print(self.thresholds[i])
        else:
            for i in range(self.file_num):
                self.thresholds[i] = ti.floor(i / (self.file_num - 1) * self.sample_num)
                print(self.thresholds[i])

    def output_filename(self, ins='output'):
        out_dir = self.file_path.parent / 'output'
        if not (out_dir.exists() and out_dir.is_dir()):
            out_dir.mkdir()
        return str(out_dir / (self.name_base + ins + self.file_path.suffix))

    def gen_file_list(self):
        lst = []
        for i in range(self.file_num):
            name = str(self.file_path.parent / f'{self.name_base}_{i + 1}{self.file_path.suffix}')
            lst.append(name)
        return lst

    def blur_mix_sdf(self):
        for k, sdf in enumerate(self.file_name_list):
            self.sdf_2d.reset(sdf)
            self.sdf_2d.gen_udf_w_h()
            self.create_sdf_buffer(k)
        self.calc_thresholds()
        self.blur_mix(self.thresholds)
        imageio.imwrite(self.output_filename('_blur_mix'), self.output_pic.to_numpy())

    def create_sdf_buffer(self, k):
        self.copy_sdf_buffer(k, self.sdf_2d.bit_pic_white, self.sdf_2d.bit_pic_black,
                             self.sdf_2d.num)

    @ti.kernel
    def copy_sdf_buffer(self, k: ti.i32, bit_pic_w: ti.template(), bit_pic_b: ti.template(), n: ti.i32):
        for i, j in ti.ndrange(self.width, self.height):
            self.sdf_buffer[i, j, k] = ti.sqrt(bit_pic_w[n, i, j][2]) - ti.sqrt(bit_pic_b[n, i, j][2])

    @ti.func
    def cal_grey_value(self, dis1, dis2, interval_l, interval_r):
        value = vec3(0)
        interval_len = interval_r - interval_l - 1
        if dis1 < -eps and dis2 < -eps:
            value = vec3(255) * (interval_len + 1)
        elif dis1 > 0.0 and dis2 > 0.0:
            pass
        else:
            res = 0
            for n in range(interval_l, interval_r):
                mix = (n - interval_l) / interval_len
                if (1 - mix) * dis1 + mix * dis2 < -eps:
                    res += 255
            value = vec3(res)
        return value

    @ti.kernel
    def blur_mix(self, thresholds: ti.template()):
        for i, j in self.output_pic:

            for k in range(self.file_num - 1):
                self.output_pic[i, j] += self.cal_grey_value(self.sdf_buffer[i, j, k], self.sdf_buffer[i, j, k + 1],
                                                             thresholds[k], thresholds[k + 1])
            self.output_pic[i, j] = int(self.output_pic[i, j] / self.sample_num)


def mask2sdf(mask_path, save_path):
    label = imageio.imread_v2(mask_path)
    label[label == 1] = 255
    label[label != 255] = 0

    mySDF2D = SDF2D(label, mask_path)
    mySDF2D.mask2sdf(output=True, save_path=save_path)


import tifffile as tif
import numpy as np


def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Preprocessing for microscopy image segmentation (from 3class to sdf)', add_help=False)
    parser.add_argument('-i', '--input_path', default='./data/Train_Pre_3class', type=str,
                        help='3class data path; subfolders: images, labels')
    parser.add_argument("-o", '--output_path', default='./data/Train_Pre_sdf', type=str, help='preprocessing data path')
    args = parser.parse_args()

    source_path = args.input_path
    target_path = args.output_path

    img_path = join(source_path, 'images')
    gt_path = join(source_path, 'labels')

    img_names = sorted(os.listdir(img_path))
    gt_names = [img_name.split('.')[0] + '_label.png' for img_name in img_names]

    pre_img_path = join(target_path, 'images')
    pre_gt_path = join(target_path, 'labels')
    os.makedirs(pre_img_path, exist_ok=True)
    os.makedirs(pre_gt_path, exist_ok=True)

    for img_name, gt_name in zip(tqdm(img_names), gt_names):
        save_img = join(target_path, 'images', img_name.split('.')[0] + '.png')
        if os.path.exists(save_img):
            continue

        img_data = imageio.imread(join(img_path, img_name))
        gt_data = imageio.imread(join(gt_path, gt_name))

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

        mask2sdf(join(gt_path, gt_name), save_path=join(target_path, 'labels', gt_name.split('.')[0] + '.png'))
        imageio.imwrite(join(target_path, 'images', img_name.split('.')[0] + '.png'), pre_img_data.astype(np.uint8))
