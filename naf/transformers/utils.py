import random
import re
from copy import deepcopy
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.utils import no_collation
from monai.metrics import CumulativeIterationMetric, do_metric_reduction
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
from monai.utils import MetricReduction
from numba import jit
from scipy.optimize import linear_sum_assignment
from skimage import segmentation, measure, morphology


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
            y: ground truth to compute mean dice metric. It must be one-hot format and first dim is batch.
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


from scipy.ndimage import binary_erosion, grey_dilation, generate_binary_structure, find_objects, gaussian_filter


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

    avg_size = sum(sizes[max_id - 10: max_id]) / lll

    if avg_size >= max_size:
        label[label > 0] = 1
        label = binary_erosion(label, iterations=10)
        label = measure.label(morphology.remove_small_holes(label, area_threshold=16), background=0)
        label = grey_dilation(label, size=(20, 20))

    return label
