from transforms.utils import swap_fft_lowpass_for, fft_highpass_filter, fft_mask_mag
import tifffile as tif
import numpy as np
import cv2
import glob

files = glob.glob("test_datas/*")


for f in files:
    img1 = tif.imread(f)
    f
    cv2.imwrite("swap_img.png", img1)



if __name__ == '__main__':

    img1_path = "test_datas/9517.tiff"
    img2_path = "test_datas/8116.tiff"


    img1 = tif.imread(img1_path)
    img2 = tif.imread(img2_path)
    white_img = np.ones_like(img1) * 128

    # print(img1.max(), img1.min())

    # r = np.clip(fft_highpass_filter(img1[..., 0], 0.01, fill_value=2), 0, 255)
    # g = np.clip(fft_highpass_filter(img1[..., 1], 0.01, fill_value=2), 0, 255)
    # b = np.clip(fft_highpass_filter(img1[..., 2], 0.01, fill_value=2), 0, 255)

    # r = np.clip(swap_fft_lowpass_for(img1[..., 0], white_img[..., 0], 0.99), 0, 255)
    # g = np.clip(swap_fft_lowpass_for(img1[..., 1], white_img[..., 0], 0.99), 0, 255)
    # b = np.clip(swap_fft_lowpass_for(img1[..., 2], white_img[..., 0], 0.99), 0, 255)

    r = np.clip(fft_mask_mag(img1[..., 0]), 0, 255)
    g = np.clip(fft_mask_mag(img1[..., 1]), 0, 255)
    b = np.clip(fft_mask_mag(img1[..., 2]), 0, 255)

    res = np.stack([r, g, b], axis=2)

    print(res.shape)

    # tif.imwrite("swap_img_01.tiff", res)
    cv2.imwrite("swap_img_1.png", res)
