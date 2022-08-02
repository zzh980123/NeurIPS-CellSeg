import numpy as np
import imageio
import tifffile

#path = '../data/Train_Pre_3class/images/cell_00100.png'
path = '/media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/cell_00077.tif'
#test_img = imageio.imread_v2(path)
test_img = tifffile.imread(path)


from scipy.fft import fft,fft2,fftshift, ifftshift, ifft2
import scipy.signal.windows as windows
import scipy.interpolate as interpol
from skimage.transform import resize
import math
mat = np.array([0.2126,0.7152,0.0722])

if len(test_img.shape) == 3:
    test_img = test_img @ mat.T
test_img_resize = test_img.copy()

test_img_resize = resize(test_img_resize, (min(test_img.shape[0],test_img.shape[1]), min(test_img.shape[0],test_img.shape[1])))
# test_img_resize.resize(min(test_img.shape[0],test_img.shape[1]), min(test_img.shape[0],test_img.shape[1]))

window1d = windows.hann(test_img_resize.shape[0])
window2d = np.sqrt(np.outer(window1d,window1d))

test_img_resize = test_img_resize * window2d
res: np.ndarray = np.fft.fft2(test_img_resize, axes=(0,1))
# res_real = np.real(res)
# res_complex = res.imag
#res_real[0,0] = res_real[0,0] /2
#res_real = res_real / (res_real.size()/2)

# res_real_center = np.log(1+np.abs(fftshift(res_real)))
shift_ = np.fft.fftshift(res, axes=(0,1))
mask = np.ones_like(shift_)

size_ = mask.shape[0]

remove_precent = 0.03
rm = int(size_ * remove_precent)
mask[size_ // 2 - rm: size_ // 2+ rm, size_ // 2 - rm: size_ // 2 + rm] = 0
# mask[size_ // 2 - 2: size_ // 2 + 2, size_ // 2 - 2: size_ // 2 + 2] = 1
shift_ *= mask


res = np.fft.ifftshift(shift_, axes=(0,1))
# res.real = res_real


res = np.fft.ifft2(res, axes=(0,1))


imageio.imwrite("./res_real.png", np.abs(res))
imageio.imwrite("./res.png", test_img_resize)
