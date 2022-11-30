import json
from logging import log
import math
import random
from os import PathLike
from typing import Dict, Hashable, Mapping, Union, Tuple, Sequence, Any, List, Optional

import numpy as np
import staintools
import tifffile
import torch
from matplotlib.colors import Normalize
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data import ImageReader, is_supported_format
from monai.data.image_reader import _copy_compatible_dict, _stack_images
from monai.metrics import CumulativeIterationMetric, do_metric_reduction
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.transforms.utility.array import (
    AddChannel,
)
from monai.utils import MetricReduction, ensure_tuple, TransformBackends
from numba import jit
from scipy.optimize import linear_sum_assignment
from skimage import segmentation, measure, morphology
import cv2
import matplotlib.cm as cm


class ConditionAddChannelFirstd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddChannel`.
    """

    backend = AddChannel.backend

    def __init__(self, keys: KeysCollection, target_dims, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.adder = AddChannel()
        self.target_dims = target_dims

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            if len(d[key].shape) == self.target_dims:
                d[key] = self.adder(d[key])
        return d


class ConditionAddChannelLastd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddChannel`.
    """

    backend = AddChannel.backend

    def __init__(self, keys: KeysCollection, target_dims, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.target_dims = target_dims

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            if len(d[key].shape) == self.target_dims:
                d[key] = d[key][..., None]

        return d


class ConditionChannelNumberd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddChannel`.
    """

    backend = AddChannel.backend

    def __init__(self, keys: KeysCollection, target_dim, channel_num, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.target_dims = target_dim
        self.channel_num = channel_num

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):

            if d[key].shape[self.target_dims] == 1:
                rp = np.repeat if isinstance(d[key], np.ndarray) else torch.repeat_interleave
                d[key] = rp(d[key], self.channel_num, self.target_dims)
            elif d[key].shape[self.target_dims] > self.channel_num:
                raise RuntimeWarning("Please reduce the channel dimensions first.")
        d['img_meta_dict']['original_channel_dim'] = -1
        return d


class ConditionSliceChanneld(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddChannel`.
    """

    def __init__(self, keys: KeysCollection, slice_start, slice_end, channel_dim=0, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.slice = (slice_start, slice_end)
        self.channel_dim = channel_dim

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            len_ = d[key].shape[self.channel_dim]

            if self.slice[1] < 0:
                self.channel_dim = d[key].shape[self.channel_dim] + self.slice[1] + 1

            if len_ >= self.slice[1]:
                d[key] = d[key][(slice(None),) * self.slice[0] + (slice(self.slice[0], self.slice[1]),) + (slice(None),) * (len_ - self.slice[1])]
        return d


class CenterCropByPercentd(MapTransform):
    def __init__(self, keys: KeysCollection, percent=(0.25, 0.25), allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.percent = percent
        for i in percent:
            assert 1 >= i > 0

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            datas = d[key]
            shape = datas.shape
            assert len(shape) == 3
            C, H, W = shape
            h = int(H * self.percent[0])
            w = int(W * self.percent[1])
            start_h = int((H - h) * 0.5)
            start_w = int((W - w) * 0.5)
            d[key] = datas[:, start_h: start_h + h, start_w: start_w + w]

        return d


class LoadJson2Tensor(MapTransform):

    def __init__(self, keys: KeysCollection, func, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.func = func

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            # to gray
            json_path = d[key]
            assert isinstance(json_path, str)

            with open(json_path, 'r') as f:
                res = json.load(f)
                tensor = self.func(res)
                d[key] = tensor

        return d


class TiffReader2(ImageReader):

    def __init__(self, channel_dim: Optional[int] = None, transpose=False, **kwargs):
        super().__init__()
        self.channel_dim = channel_dim
        self.kwargs = kwargs
        self.transpose = transpose

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        """
        Verify whether the specified `filename` is supported by the current reader.
        This method should return True if the reader is able to read the format suggested by the
        `filename`.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.

        """
        raise is_supported_format(filename, ["tif", "tiff"])

    def read(self, data: Union[Sequence[PathLike], PathLike], **kwargs) -> Union[Sequence[Any], Any]:
        """
        Read image data from specified file or files.
        Note that it returns a data object or a sequence of data objects.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for actual `read` API of 3rd party libs.

        """
        img_: List[np.ndarray] = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for name in filenames:
            # we transpose the width dimension and height dimension...
            # if the label has vector-field, it is needed to fix the direction of the flows
            img = tifffile.imread(name, **kwargs_)
            if self.transpose:
                img = img.transpose(0, 2, 1)
            img_.append(img)
        return img_ if len(img_) > 1 else img_[0]

    def get_data(self, img) -> Tuple[np.ndarray, Dict]:
        """
        Extract data array and metadata from loaded image and return them.
        This function must return two objects, the first is a numpy array of image data,
        the second is a dictionary of metadata.

        Args:
            img: an image object loaded from an image file or a list of image objects.

        """
        img_array: List[np.ndarray] = []
        compatible_meta: Dict = {}
        if isinstance(img, np.ndarray):
            img = (img,)

        for i in ensure_tuple(img):
            header = {}
            if isinstance(i, np.ndarray):
                # if `channel_dim` is None, can not detect the channel dim, use all the dims as spatial_shape
                spatial_shape = np.asarray(i.shape)
                if isinstance(self.channel_dim, int):
                    spatial_shape = np.delete(spatial_shape, self.channel_dim)
                header["spatial_shape"] = spatial_shape
                # header["width"] = spatial_shape[0]
                # header["height"] = spatial_shape[1]

            img_array.append(i)
            header["original_channel_dim"] = self.channel_dim if isinstance(self.channel_dim, int) else "no_channel"
            _copy_compatible_dict(header, compatible_meta)
        return _stack_images(img_array, compatible_meta), compatible_meta


class Flow2dTransposeFixd(MapTransform):

    def __init__(self, keys: KeysCollection, flow_dim_start=0, flow_dim_end=2, allow_missing_keys: bool = False):
        """
        The flow may embed into the Channel-Dimension we defined the C at the last dimension of the tensor.
        Users should tell witch the flow start and end.
        For example, the label's dims are: (H, W, C), and the flow is in the channel dimension start at 2 and end at 3:

        flow = input[:, :, 2:4]

        The TiffReader2 will cause the H and W be swapped and the flow's direction being shuffled.
        This function will recover the direction of flow by rotation and flip.

        @rtype: Fixed flows.
        """
        super().__init__(keys, allow_missing_keys)
        self.flow_slice = slice(flow_dim_start, flow_dim_end)
        theta = math.pi / 2 * 3
        self.rotate_matrix = np.array([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)],
        ])

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        for key in self.key_iterator(d):
            flow_ = d[key][self.flow_slice]
            # flip y axis
            flow_[:1] *= -1
            C, H, W = flow_.shape
            # rotate vector in the field
            flow_ = self.rotate_matrix @ flow_.reshape(C, H * W)
            d[key][self.flow_slice] = flow_.reshape(C, H, W)

        return d


class Flow2dRoatation90Fixd(MapTransform):

    def __init__(self, keys: KeysCollection, flow_dim_start=0, flow_dim_end=2, allow_missing_keys: bool = False):
        """

        @rtype: Fixed flows.
        """
        super().__init__(keys, allow_missing_keys)
        self.flow_slice = slice(flow_dim_start, flow_dim_end)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):

            all_transforms_flag = f"{key}_transforms"
            rank_k = -1
            if all_transforms_flag in d:
                transforms = d[all_transforms_flag]
                for t in transforms:
                    if t["class"] == "RandRotate90d" and t["do_transforms"]:
                        if rank_k == -1: rank_k = 0
                        rank_k = (rank_k + t["extra_info"]["rand_k"]) % 4

            if rank_k > 0:
                flow_ = d[key][self.flow_slice]
                # rotate vector in the field
                d[key][self.flow_slice] = rotate_flow(flow_, rank_k=rank_k)

        return d


def get_rotate_matrix(theta, is_tensor=False):
    rotate_matrix = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)],
    ])

    return rotate_matrix


def rotate_flow(flow, rank_k=1, theta=None):
    # print(flow.shape)
    B = 1
    if len(flow.shape) == 4:
        B, C, H, W = flow.shape
    elif len(flow.shape) == 3:
        C, H, W = flow.shape

    if theta is None:
        rm = get_rotate_matrix(math.pi / 2 * rank_k)
    else:
        rm = get_rotate_matrix(theta=theta)

    if isinstance(flow, torch.Tensor):
        rm = torch.from_numpy(rm).float()

    flow = (rm @ flow.reshape(C, H * W)).reshape(C, H, W)

    return flow


class Flow2dRoatateFixd(MapTransform):

    def __init__(self, keys: KeysCollection, flow_dim_start=0, flow_dim_end=2, allow_missing_keys: bool = False):
        """

        @rtype: Fixed flows.
        """
        super().__init__(keys, allow_missing_keys)
        self.flow_slice = slice(flow_dim_start, flow_dim_end)

    def get_rotate_matrix(self, theta):
        rotate_matrix = np.array([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)],
        ])

        return rotate_matrix

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            assert d[key][self.flow_slice].ndim == 3

            all_transforms_flag = f"{key}_transforms"
            rot_mat = np.eye(2)
            if all_transforms_flag in d:
                transforms = d[all_transforms_flag]
                for t in transforms:
                    if t["class"] == "RandRotated" and t["do_transforms"]:
                        _t = t["extra_info"]["rot_mat"]
                        inv_rot_mat = np.linalg.inv(_t)

                        rot_mat = rot_mat @ inv_rot_mat[:2, :2]

            flow_ = d[key][self.flow_slice]
            # rotate vector in the field
            C, H, W = flow_.shape
            flow_ = rot_mat @ flow_.reshape(C, H * W)
            d[key][self.flow_slice] = flow_.reshape(C, H, W)

        return d


class Flow2dFlipFixd(MapTransform):

    def __init__(self, keys: KeysCollection, flow_dim_start=0, flow_dim_end=2, allow_missing_keys: bool = False):
        """

        @rtype: Fixed flows.
        """
        super().__init__(keys, allow_missing_keys)
        self.flow_slice = slice(flow_dim_start, flow_dim_end)
        # theta = 0
        # self.rotate_matrix = np.array([
        #     [math.cos(theta), -math.sin(theta)],
        #     [math.sin(theta), math.cos(theta)],
        # ])

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):

            all_transforms_flag = f"{key}_transforms"
            axis = None
            if all_transforms_flag in d:
                transforms = d[all_transforms_flag]
                for t in transforms:
                    if t["class"] == "RandAxisFlipd" and t["do_transforms"]:
                        axis = t["extra_info"]["axis"]
                        break

            if axis is not None:
                flow_ = d[key][self.flow_slice]
                # rotate vector in the field
                if axis == 0:
                    flow_[:1] *= -1
                elif axis == 1:
                    flow_[1:] *= -1
                else:
                    raise RuntimeError("Only support 2d flow.")
                C, H, W = flow_.shape

                flow_ = flow_.reshape(C, H * W)
                d[key][self.flow_slice] = flow_.reshape(C, H, W)

        return d


class StainNetNormalized(MapTransform):

    def __init__(self, keys: KeysCollection, checkpoint="./augment/stain_augment/StainNet/checkpoints/aligned_cytopathology_dataset/StainNet-3x0_best_psnr_layer3_ch32.pth",
                 device="cpu", allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        from augment.stain_augment.StainNet.models import StainNet
        stain_model = StainNet().to(device)

        stain_model.load_state_dict(torch.load(checkpoint))
        stain_model.eval()
        stain_model.requires_grad_(False)
        self.normalizer = stain_model

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        for key in self.key_iterator(d):
            if not isinstance(d[key], torch.Tensor):
                d[key] = torch.from_numpy(d[key])
            d[key] = self.normalizer(d[key])

        return d


class StainNormalized(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.normalizer = staintools.ReinhardColorNormalizer()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.normalizer.transform(d[key])

        return d


# Node that the mixcut op will store a mask of the cut area, if you transform the original images, the mixcut_mask should be transformed too,
# so we suggest to use the operation at the end of the transforms' chain.
# only 2D images with channel first support.
class RandCutoutd(RandomizableTransform, MapTransform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False, prob=0.5, lam=0.01, num=16, do_transform: bool = True):
        RandomizableTransform.__init__(self, prob)
        MapTransform.__init__(self, keys)
        self.lam = lam
        self.num = num

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Mapping[Hashable, NdarrayOrTensor]:
        self.randomize(data)
        d = dict(data)
        mask = 1

        bbox_list = []
        flag = False
        for key in self.key_iterator(d):

            if not flag:
                bbox_list = self.cutout(self.num, d[key].shape, self.lam)
                if isinstance(d[key], np.ndarray):
                    mask = np.ones(d[key][:1].shape)
                elif isinstance(d[key], torch.Tensor):
                    mask = torch.ones(d[key][:1].shape)
                for bbx1, bbx2, bby1, bby2 in bbox_list:
                    mask[..., bbx1: bbx2, bby1: bby2] = 0
                flag = True
            if not self._do_transform:
                # We break the transfomers and set the "cutout_mask" buffer.
                break

            d[key] = d[key] * mask

        d['cutout_mask'] = mask

        return d

    def cutout(self, num, size, lam):
        res = []

        for _ in range(num):
            res.append(tuple(self._cut(size, lam)))

        return res

    def _cut(self, size, lam):
        H = size[-2]
        W = size[-1]

        cut_rat = np.sqrt(lam)
        cut_h = np.int(W * cut_rat)
        cut_w = np.int(H * cut_rat)
        cx = np.random.randint(H)
        cy = np.random.randint(W)
        bbx1 = np.clip(cx - cut_h // 2.0, 0, H)
        bbx2 = np.clip(cx + cut_h // 2.0, bbx1, H)
        bby1 = np.clip(cy - cut_w // 2.0, 0, W)
        bby2 = np.clip(cy + cut_w // 2.0, bby1, H)

        return int(bby1), int(bby2), int(bbx1), int(bbx2)


class ColorJitterd(MapTransform):
    def __init__(self, keys: KeysCollection, brightness=.1, contrast=.1, saturation=.1, hue=.2, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        from torchvision.transforms.transforms import ColorJitter
        self.colorconvert = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        for key in self.key_iterator(d):
            if isinstance(d[key], np.ndarray):
                d[key] = torch.from_numpy(d[key])
            d[key] = self.colorconvert(d[key])

        return d


class FFTFilterd(MapTransform):
    """
        """

    def __init__(self, keys: KeysCollection, hf_mask_percent=0, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.mat = np.array([0.2126, 0.7152, 0.0722]).reshape((3,) + (1,) * 2)
        self.hf_mask_percent = hf_mask_percent

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            # to gray

            rgb_img = d[key]

            if not isinstance(rgb_img, np.ndarray):
                raise RuntimeError("FFT transform only support numpy array.")

            if rgb_img.shape[0] == 3 and rgb_img.ndim == 3:
                gray_img = np.sum(rgb_img * self.mat, keepdims=True, axis=0)
            else:
                gray_img = rgb_img

            res = np.fft.fft2(gray_img, axes=(0, 1))

            shift_ = np.fft.fftshift(res, axes=(0, 1))
            mask = np.ones_like(shift_)

            remove_percent = self.hf_mask_percent
            lf = False
            if remove_percent < 0:
                remove_percent = -remove_percent

            size_ = mask.shape[0]

            rm = int(size_ * remove_percent)
            mask[size_ // 2 - rm: size_ // 2 + rm, size_ // 2 - rm: size_ // 2 + rm] = 0
            if remove_percent > 0 and not lf:
                shift_ *= mask
            elif lf:
                shift_ *= (1 - mask)
            else:
                pass

            res = np.fft.ifftshift(shift_, axes=(0, 1))
            res = np.fft.ifft2(res, axes=(0, 1))

            d[key] = np.abs(res)

        return d


def fft_highpass_filter(rgb_img, hf_mask_percent=0.1, fill_value=0):
    mat = np.array([0.2126, 0.7152, 0.0722]).reshape((3,) + (1,) * 2)
    if not isinstance(rgb_img, np.ndarray):
        raise RuntimeError("FFT transform only support numpy array.")

    if rgb_img.shape[0] == 3 and rgb_img.ndim == 3:
        gray_img = np.sum(rgb_img * mat, keepdims=True, axis=0)
    else:
        gray_img = rgb_img

    res = np.fft.fft2(gray_img, axes=(0, 1))

    shift_ = np.fft.fftshift(res, axes=(0, 1))
    mask = np.ones_like(shift_)

    remove_percent = hf_mask_percent
    lf = False
    if remove_percent < 0:
        remove_percent = -remove_percent
        lf = True

    size_ = mask.shape[0]

    rm = int(size_ * remove_percent)
    mask[size_ // 2 - rm: size_ // 2 + rm, size_ // 2 - rm: size_ // 2 + rm] = 0
    if remove_percent > 0 and not lf:
        shift_ *= mask + (1 - mask) * fill_value
    elif lf:
        shift_ *= (1 - mask) + mask * fill_value
    else:
        pass

    res = np.fft.ifftshift(shift_, axes=(0, 1))
    res = np.fft.ifft2(res, axes=(0, 1))

    return np.abs(res)


def fft_mask_mag(rgb_img, hf_mask_percent=0.1):
    mat = np.array([0.2126, 0.7152, 0.0722]).reshape((3,) + (1,) * 2)
    if not isinstance(rgb_img, np.ndarray):
        raise RuntimeError("FFT transform only support numpy array.")

    if rgb_img.shape[0] == 3 and rgb_img.ndim == 3:
        gray_img = np.sum(rgb_img * mat, keepdims=True, axis=0)
    else:
        gray_img = rgb_img

    res = np.fft.fft2(gray_img, axes=(0, 1))

    shift_ = np.fft.fftshift(res, axes=(0, 1))

    phase_spectrum = np.angle(shift_)
    mag_spectrum = np.abs(shift_)

    # mask = np.ones_like(shift_)

    # remove_percent = hf_mask_percent
    # lf = False
    # if remove_percent < 0:
    #     remove_percent = -remove_percent
    #     lf = True
    #
    # size_ = mask.shape[0]
    #
    # rm = int(size_ * remove_percent)
    # mask[size_ // 2 - rm: size_ // 2 + rm, size_ // 2 - rm: size_ // 2 + rm] = 0
    # if remove_percent > 0 and not lf:
    #     shift_ *= mask
    # elif lf:
    #     shift_ *= 1 - mask
    # else:
    #     pass
    image_combine = mag_spectrum * np.e ** (1j * phase_spectrum)

    res = np.fft.ifftshift(image_combine, axes=(0, 1))
    res = np.fft.ifft2(res, axes=(0, 1))

    return np.abs(res)


def swap_fft_lowpass_for(rgb_img1, rgb_img2, hf_mask_percent=0.01):
    mat = np.array([0.2126, 0.7152, 0.0722]).reshape((3,) + (1,) * 2)
    if not isinstance(rgb_img1, np.ndarray) or not isinstance(rgb_img2, np.ndarray):
        raise RuntimeError("FFT transform only support numpy array.")

    if rgb_img1.shape[0] == 3 and rgb_img1.ndim == 3:
        gray_img1 = np.sum(rgb_img1 * mat, keepdims=True, axis=0)
    else:
        gray_img1 = rgb_img1

    if rgb_img2.shape[0] == 3 and rgb_img2.ndim == 3:
        gray_img2 = np.sum(rgb_img2 * mat, keepdims=True, axis=0)
    else:
        gray_img2 = rgb_img2

    assert rgb_img1.shape == rgb_img2.shape

    res1 = np.fft.fft2(gray_img1, axes=(0, 1))
    res2 = np.fft.fft2(gray_img2, axes=(0, 1))

    shift_1 = np.fft.fftshift(res1, axes=(0, 1))
    shift_2 = np.fft.fftshift(res2, axes=(0, 1))
    mask = np.ones_like(shift_1)

    remove_percent = hf_mask_percent
    lf = False
    if remove_percent < 0:
        remove_percent = -remove_percent

    size_ = mask.shape[0]

    rm = int(size_ * remove_percent)
    mask[size_ // 2 - rm: size_ // 2 + rm, size_ // 2 - rm: size_ // 2 + rm] = 0
    mask2 = 1 - mask

    if remove_percent > 0 and not lf:
        shift_1 *= mask
        shift_2 *= mask2
        shift_ = shift_1 + shift_2
    elif lf:
        shift_1 *= mask2
        shift_2 *= mask
        shift_ = shift_1 + shift_2
    else:
        raise RuntimeError("Remove percent should be 0~1.")

    res = np.fft.ifftshift(shift_, axes=(0, 1))
    res = np.fft.ifft2(res, axes=(0, 1))

    return np.abs(res)


import imgaug.augmenters.color as color_aug


class RandBrightnessd(RandomizableTransform, MapTransform, InvertibleTransform):

    def __init__(self, keys: KeysCollection, add=(-20, 20), prob=1.0, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob, do_transform=True)
        self.changer = color_aug.AddToBrightness(add=add)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        if random.random() > self.prob:
            return d

        for key in self.key_iterator(d):
            data = d[key]
            assert data.ndim == 3
            # colorspace transform need unsigned int
            # opencv need the channel at the last dimension
            flag = False
            if 3 in data.shape and data.shape[0] == 3:
                data = np.transpose(data, axes=(1, 2, 0))
                flag = True
            res = self.changer.augment_image(data.astype(np.uint8))
            if flag:
                res = np.transpose(res, axes=(2, 0, 1))
            d[key] = res

        return d


class RandHueAndSaturationd(RandomizableTransform, MapTransform, InvertibleTransform):

    def __init__(self, keys: KeysCollection, prob=1.0, add=(-30, 30), allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob, do_transform=True)
        self.changer = color_aug.AddToHueAndSaturation(value_hue=add, per_channel=True)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        if random.random() > self.prob:
            return d

        for key in self.key_iterator(d):
            data = d[key]
            assert data.ndim == 3
            # colorspace transform need unsigned int
            # opencv need the channel at the last dimension
            flag = False
            if 3 in data.shape and data.shape[0] == 3:
                data = np.transpose(data, axes=(1, 2, 0))
                flag = True
            res = self.changer.augment_image(data.astype(np.uint8))
            if flag:
                res = np.transpose(res, axes=(2, 0, 1))
            d[key] = res
        return d


class RandInversed(RandomizableTransform, MapTransform, InvertibleTransform):

    def __init__(self, keys: KeysCollection, prob=1.0, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob, do_transform=True)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        if random.random() > self.prob:
            return d

        for key in self.key_iterator(d):
            data = d[key]
            d[key] = data.max() - data
        return d


def _intersection_over_union(masks_true, masks_pred):
    """ intersection over union of all mask pairs

    Parameters
    ------------

    masks_true: ND-array, int
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels
    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


@jit(nopython=True)
def _label_overlap(x, y):
    """ fast function to get pixel overlaps between masks in x and y

    Parameters
    ------------

    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]

    """
    x = x.ravel()
    y = y.ravel()

    # preallocate a 'contact map' matrix
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)

    # loop over the labels in x and add to the corresponding
    # overlap entry. If label A in x and label B in y share P
    # pixels, then the resulting overlap is P
    # len(x)=len(y), the number of pixels in the whole image
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap


def _true_positive(iou, th):
    """ true positive at threshold th

    Parameters
    ------------

    iou: float, ND-array
        array of IOU pairs
    th: float
        threshold on IOU for positive label

    Returns
    ------------

    tp: float
        number of true positives at threshold
    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    return tp


def eval_tp_fp_fn(masks_true, masks_pred, threshold=0.5):
    num_inst_gt = np.max(masks_true)
    num_inst_seg = np.max(masks_pred)
    if num_inst_seg > 0:
        iou = _intersection_over_union(masks_true, masks_pred)[1:, 1:]
        # for k,th in enumerate(threshold):
        tp = _true_positive(iou, threshold)
        fp = num_inst_seg - tp
        fn = num_inst_gt - tp
    else:
        print('No segmentation results!')
        tp = 0
        fp = 0
        fn = 0

    return tp, fp, fn


def remove_boundary_cells(mask):
    W, H = mask.shape
    bd = np.ones((W, H))
    bd[2:W - 2, 2:H - 2] = 0
    bd_cells = np.unique(mask * bd)
    for i in bd_cells[1:]:
        mask[mask == i] = 0
    new_label, _, _ = segmentation.relabel_sequential(mask)
    return new_label


class CellF1Metric(CumulativeIterationMetric):

    def __init__(self,
                 get_not_nans: bool = False,
                 reduction='mean_batch'):
        super().__init__()
        self.reduction = reduction
        self.get_not_nans = get_not_nans

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):  # type: ignore
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
                should be binarized.
            y: ground truth to compute mean f1 metric. It must be one-hot format and first dim is batch.
                The values should be binarized.

        Raises:
            ValueError: when `y` is not a binarized tensor.
            ValueError: when `y_pred` has less than three dimensions.
        """

        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError(f"y_pred should have at least 3 dimensions (batch, channel, spatial), got {dims}.")
        # compute dice (BxC) for each channel for each batch
        return self.compute_F1_score(
            seg=y_pred, gt=y
        )

    def aggregate(self, reduction: Union[MetricReduction, str, None] = None):  # type: ignore
        """
        Execute reduction logic for the output of `compute_meandice`.

        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.

        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f

    def compute_F1_score(self, gt, seg):
        # The batch size and the channel number should be one
        assert gt.shape[0] == seg.shape[0] == gt.shape[1] == seg.shape[1] == 1

        gt = gt[0][0].cpu().numpy()
        seg = seg[0][0].cpu().numpy()

        # Score the cases
        # do not consider cells on the boundaries during evaluation
        if np.prod(gt.shape) < 25000000:
            gt = remove_boundary_cells(gt.astype(np.int32))
            seg = remove_boundary_cells(seg.astype(np.int32))
            tp, fp, fn = eval_tp_fp_fn(gt, seg, threshold=0.5)
        else:  # for large images (>5000x5000), the F1 score is computed by a patch-based way
            H, W = gt.shape
            roi_size = 2000

            if H % roi_size != 0:
                n_H = H // roi_size + 1
                new_H = roi_size * n_H
            else:
                n_H = H // roi_size
                new_H = H

            if W % roi_size != 0:
                n_W = W // roi_size + 1
                new_W = roi_size * n_W
            else:
                n_W = W // roi_size
                new_W = W

            gt_pad = np.zeros((new_H, new_W), dtype=gt.dtype)
            seg_pad = np.zeros((new_H, new_W), dtype=gt.dtype)
            gt_pad[:H, :W] = gt
            seg_pad[:H, :W] = seg

            tp = 0
            fp = 0
            fn = 0
            for i in range(n_H):
                for j in range(n_W):
                    gt_roi = remove_boundary_cells(gt_pad[roi_size * i:roi_size * (i + 1), roi_size * j:roi_size * (j + 1)])
                    seg_roi = remove_boundary_cells(seg_pad[roi_size * i:roi_size * (i + 1), roi_size * j:roi_size * (j + 1)])
                    tp_i, fp_i, fn_i = eval_tp_fp_fn(gt_roi, seg_roi, threshold=0.5)
                    tp += tp_i
                    fp += fp_i
                    fn += fn_i

        if tp == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)

        return torch.tensor([[f1, precision, recall]], dtype=torch.float32)


def sem2ins_label(outputs, labels_onehot, dim=1):
    assert isinstance(outputs[0], torch.Tensor) and isinstance(labels_onehot[0], torch.Tensor)

    outputs_pred_np = outputs[0][dim].cpu().numpy()
    outputs_label_np = labels_onehot[0][dim].cpu().numpy()
    # convert probability map to binary mask and apply morphological postprocessing
    outputs_pred_mask = measure.label(morphology.remove_small_objects(morphology.remove_small_holes(outputs_pred_np > 0.5), 16))
    outputs_label_mask = measure.label(morphology.remove_small_objects(morphology.remove_small_holes(outputs_label_np > 0.5), 16))

    # convert back to tensor for metric computing
    outputs_pred_mask = torch.from_numpy(outputs_pred_mask[None, None])
    outputs_label_mask = torch.from_numpy(outputs_label_mask[None, None])

    return outputs_pred_mask, outputs_label_mask


# Obtained from cellpose: https://github.com/MouseLand/cellpose/blob/91dd7abce332a30ead85e10e7c244bfa876c9a2d/cellpose/utils.py#L325
def get_masks_unet(output, cell_threshold=0, boundary_threshold=0):
    """ create masks using cell probability and cell boundary """
    cells = (output[..., 1] - output[..., 0]) > cell_threshold
    selem = generate_binary_structure(cells.ndim, connectivity=1)
    labels, nlabels = measure.label(cells, selem)

    if output.shape[-1] > 2:
        slices = find_objects(labels)
        dists = 10000 * np.ones(labels.shape, np.float32)
        mins = np.zeros(labels.shape, np.int32)
        borders = np.logical_and(~(labels > 0), output[..., 2] > boundary_threshold)
        pad = 10
        for i, slc in enumerate(slices):
            if slc is not None:
                slc_pad = tuple([slice(max(0, sli.start - pad), min(labels.shape[j], sli.stop + pad))
                                 for j, sli in enumerate(slc)])
                msk = (labels[slc_pad] == (i + 1)).astype(np.float32)
                msk = 1 - gaussian_filter(msk, 5)
                dists[slc_pad] = np.minimum(dists[slc_pad], msk)
                mins[slc_pad][dists[slc_pad] == msk] = (i + 1)
        labels[labels == 0] = borders[labels == 0] * mins[labels == 0]

    masks = labels
    shape0 = masks.shape
    _, masks = np.unique(masks, return_inverse=True)
    masks = np.reshape(masks, shape0)
    return masks


from scipy.ndimage import binary_erosion, grey_dilation, generate_binary_structure, find_objects, gaussian_filter, binary_dilation


def post_process(label, max_size=60 * 60):
    ids = np.unique(label)

    min_id = 1
    max_id = min(20, ids.max()) + 1
    # label[label > 0] = 1
    sizes = []
    for i in range(min_id, max_id):
        sizes.append((label == i).sum())

    sizes.sort()

    lll = len(sizes[max_id - 10:max_id])

    avg_size = sum(sizes[max_id - 10: max_id]) / max(lll, 1)

    if avg_size >= max_size:
        label[label > 0] = 1
        back_label = 1 - label
        iter_ = max(int(avg_size ** .5) // 16, 5)
        label = ~binary_dilation(back_label, iterations=iter_)
        label = binary_erosion(label, iterations=iter_)
        label = measure.label(morphology.remove_small_holes(label, area_threshold=16), background=0)
        label = grey_dilation(label, size=(iter_ * 4, iter_ * 4))

    return label


def post_process_2(label, max_size=60 * 60):
    ids = np.unique(label)

    min_id = 1
    max_id = min(40, ids.max()) + 1
    # label[label > 0] = 1
    sizes = []
    for i in range(min_id, max_id):
        sizes.append((label == i).sum())

    sizes.sort()

    lll = len(sizes[max_id - 10:max_id])

    avg_size = sum(sizes[max_id - 10: max_id]) / max(lll, 1)

    if avg_size >= max_size:
        label[label > 0] = 1
        # back_label = 1 - label
        iter_ = min(int(avg_size ** .5) // 16, 4)
        # label = ~binary_dilation(back_label, iterations=iter_)
        label = binary_erosion(label, iterations=iter_)
        label = measure.label(morphology.remove_small_holes(label, area_threshold=16), background=0)
        label = grey_dilation(label, size=(iter_ * 2, iter_ * 2))

    return label


from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def post_process_3(label, min_distance=4, cell_prob=0.5, markers=None):
    # watershed
    label[label > cell_prob] = 1
    label[label <= cell_prob] = 0

    # label = ndi.binary_opening(label, structure=None)

    if markers is None:
        distance = ndi.distance_transform_edt(label)

        max_coords = peak_local_max(distance, labels=label, min_distance=min_distance,
                                    footprint=np.ones((2, 2)))
        local_maxima = np.zeros_like(label, dtype=bool)
        local_maxima[tuple(max_coords.T)] = True
        markers = ndi.label(local_maxima)[0]

    label = watershed(label, markers=markers, mask=label)

    # label = ndi.grey_opening(label, size=(3,3), footprint=(3, 3))

    return label


def three_classes2instance(label):
    label = label > 0
    return measure.label(label)


import matplotlib.pyplot as plt


def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    import PIL.Image as Image
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    return image


def flow(slices_in,  # the 2D slices
         titles=None,  # list of titles
         cmaps=None,  # list of colormaps
         width=15,  # width in in
         img_indexing=True,  # whether to match the image view, i.e. flip y axis
         grid=False,  # option to plot the images in a grid or a single row
         show=True,  # option to actually show the plot (plt.show())
         scale=1):  # note quiver essentially draws quiver length = 1/scale
    '''
    plot a grid of flows (2d+2 images)
    '''

    # input processing
    nb_plots = len(slices_in)
    for slice_in in slices_in:
        assert len(slice_in.shape) == 3, 'each slice has to be 3d: 2d+2 channels'
        assert slice_in.shape[-1] == 2, 'each slice has to be 3d: 2d+2 channels'

    def input_check(inputs, nb_plots, name):
        ''' change input from None/single-link '''
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [None]
        if len(inputs) == 1:
            inputs = [inputs[0] for _ in range(nb_plots)]
        return inputs

    if img_indexing:
        for si, slc in enumerate(slices_in):
            slices_in[si] = np.flipud(slc)

    titles = input_check(titles, nb_plots, 'titles')
    cmaps = input_check(cmaps, nb_plots, 'cmaps')
    scale = input_check(scale, nb_plots, 'scale')

    # figure out the number of rows and columns
    if grid:
        if isinstance(grid, bool):
            rows = np.floor(np.sqrt(nb_plots)).astype(int)
            cols = np.ceil(nb_plots / rows).astype(int)
        else:
            assert isinstance(grid, (list, tuple)), \
                "grid should either be bool or [rows,cols]"
            rows, cols = grid
    else:
        rows = 1
        cols = nb_plots

    # prepare the subplot
    fig, axs = plt.subplots(rows, cols)
    if rows == 1 and cols == 1:
        axs = [axs]

    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        # turn off axis
        ax.axis('off')

        # add titles
        if titles is not None and titles[i] is not None:
            ax.title.set_text(titles[i])

        u, v = slices_in[i][..., 0], slices_in[i][..., 1]
        colors = np.arctan2(u, v)
        colors[np.isnan(colors)] = 0
        norm = Normalize()
        norm.autoscale(colors)
        if cmaps[i] is None:
            colormap = cm.winter
        else:
            raise Exception("custom cmaps not currently implemented for plt.flow()")

        # show figure
        ax.quiver(u, v,
                  color=colormap(norm(colors).flatten()),
                  angles='xy',
                  units='xy',
                  scale=scale[i])
        ax.axis('equal')

    # clear axes that are unnecessary
    for i in range(nb_plots, col * row):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        ax.axis('off')

    # show the plots
    fig.set_size_inches(width, rows / cols * width)
    plt.tight_layout()

    if show:
        plt.show()
        plt.close(fig)

    return fig, axs, plt


"""
Obtained from cellpose
"""


def resize_image(img0, Ly=None, Lx=None, rsz=None, interpolation=cv2.INTER_LINEAR, no_channels=False):
    """ resize image for computing flows / unresize for computing dynamics
    Parameters
    -------------
    img0: ND-array
        image of size [Y x X x nchan] or [Lz x Y x X x nchan] or [Lz x Y x X]
    Ly: int, optional
    Lx: int, optional
    rsz: float, optional
        resize coefficient(s) for image; if Ly is None then rsz is used
    interpolation: cv2 interp method (optional, default cv2.INTER_LINEAR)
    Returns
    --------------
    imgs: ND-array
        image of size [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan]
    """
    if Ly is None and rsz is None:
        error_message = 'must give size to resize to or factor to use for resizing'
        print(error_message)
        raise ValueError(error_message)

    if Ly is None:
        # determine Ly and Lx using rsz
        if not isinstance(rsz, list) and not isinstance(rsz, np.ndarray):
            rsz = [rsz, rsz]
        if no_channels:
            Ly = int(img0.shape[-2] * rsz[-2])
            Lx = int(img0.shape[-1] * rsz[-1])
        else:
            Ly = int(img0.shape[-3] * rsz[-2])
            Lx = int(img0.shape[-2] * rsz[-1])

    # no_channels useful for z-stacks, sot he third dimension is not treated as a channel
    # but if this is called for grayscale images, they first become [Ly,Lx,2] so ndim=3 but
    if (img0.ndim > 2 and no_channels) or (img0.ndim == 4 and not no_channels):
        if no_channels:
            imgs = np.zeros((img0.shape[0], Ly, Lx), np.float32)
        else:
            imgs = np.zeros((img0.shape[0], Ly, Lx, img0.shape[-1]), np.float32)
        for i, img in enumerate(img0):
            imgs[i] = cv2.resize(img, (Lx, Ly), interpolation=interpolation)
    else:
        imgs = cv2.resize(img0, (Lx, Ly), interpolation=interpolation)
    return imgs


# modified to use sinebow color
def dx_to_circ(dP, transparency=False, mask=None):
    """ dP is 2 x Y x X => 'optic' flow representation

    Parameters
    -------------

    dP: 2xLyxLx array
        Flow field components [dy,dx]

    transparency: bool, default False
        magnitude of flow controls opacity, not lightness (clear background)

    mask: 2D array
        Multiplies each RGB component to suppress noise

    """

    dP = np.array(dP)
    mag = np.clip(np.sqrt(np.sum(dP ** 2, axis=0)), 0, 1.)
    angles = np.arctan2(dP[1], dP[0]) + np.pi
    a = 2
    r = ((np.cos(angles) + 1) / a)
    g = ((np.cos(angles + 2 * np.pi / 3) + 1) / a)
    b = ((np.cos(angles + 4 * np.pi / 3) + 1) / a)

    if transparency:
        im = np.stack((r, g, b, mag), axis=-1)
    else:
        im = np.stack((r * mag, g * mag, b * mag), axis=-1)

    if mask is not None and transparency and dP.shape[0] < 3:
        im[:, :, -1] *= mask

    im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    return im
