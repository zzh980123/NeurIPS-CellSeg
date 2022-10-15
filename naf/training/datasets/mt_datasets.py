import collections
from typing import Union, Sequence

from monai.data import Dataset
from monai.transforms import apply_transform
from torch.utils.data import Subset


class DualStreamDataset(Dataset):

    def __init__(self, labeled_dataset, unlabeled_dataset, weak_aug_transforms, strong_aug_transforms):
        super().__init__(labeled_dataset, transform=strong_aug_transforms)
        # self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = Dataset(unlabeled_dataset, transform=strong_aug_transforms)
        self.wt = weak_aug_transforms
        self.st = strong_aug_transforms

    def __len__(self) -> int:
        return max(len(self.data), len(self.unlabeled_dataset))

    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        data_i = self.data[index]
        no_aug = apply_transform(self.st, data_i) if self.st is not None else data_i

        return no_aug

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        labeled_idx = index % len(self.data)
        unlabeled_idx = index % len(self.unlabeled_dataset)

        labeled_data = super(DualStreamDataset, self).__getitem__(labeled_idx)
        unlabeled_data = self.unlabeled_dataset.__getitem__(unlabeled_idx)
        weak_aug_unlabeled_data = apply_transform(self.wt, unlabeled_data) if self.wt is not None else unlabeled_data

        combine_ = labeled_data
        combine_["waul_img"] = weak_aug_unlabeled_data["img"]
        combine_["ul_img"] = unlabeled_data["img"]
        combine_["cutout_mask"] = weak_aug_unlabeled_data["cutout_mask"]

        return combine_
