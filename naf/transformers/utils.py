import re
from copy import deepcopy
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.utils import no_collation
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform, Randomizable, RandomizableTransform
from monai.transforms.utility.array import (
    AddChannel,
    AddCoordinateChannels,
    AddExtremePointsChannel,
    AsChannelFirst,
    AsChannelLast,
    CastToType,
    ClassesToIndices,
    ConvertToMultiChannelBasedOnBratsClasses,
    CuCIM,
    DataStats,
    EnsureChannelFirst,
    EnsureType,
    FgBgToIndices,
    Identity,
    IntensityStats,
    LabelToMask,
    Lambda,
    MapLabelValue,
    RemoveRepeatedChannel,
    RepeatChannel,
    SimulateDelay,
    SplitDim,
    SqueezeDim,
    ToCupy,
    ToDevice,
    ToNumpy,
    ToPIL,
    TorchVision,
    ToTensor,
    Transpose,
)


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
        self.mat = np.array([0.2126, 0.7152, 0.0722]).reshape((3, ) + (1, ) * 2)
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


def fft_highpass_filter(rgb_img, hf_mask_percent=0.1):
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

    return np.abs(res)
