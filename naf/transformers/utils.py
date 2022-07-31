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
