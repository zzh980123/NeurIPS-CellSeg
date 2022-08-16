import glob
import os

import numpy as np
from monai.data import PILReader
from monai.transforms import Zoomd, Compose, LoadImaged, SaveImaged, AddChanneld, EnsureChannelFirstd, EnsureTyped

from transformers.utils import ConditionChannelNumberd, CenterCropByPercentd


def augment_small_cells(images, labels):
    zoom_transforms = Compose(
        [
            LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.uint8),
            AddChanneld(keys=["label"], allow_missing_keys=True),
            # ConditionAddChannelLastd(
            #     keys=["img"], target_dims=2, allow_missing_keys=True
            # ),
            EnsureChannelFirstd(
                keys=["img"],
            ),  # image: (3, H, W)
            ConditionChannelNumberd(
                keys=["img"], target_dim=0, channel_num=3, allow_missing_keys=True
            ),
            Zoomd(keys=["img", "label"], zoom=(0.25, 0.25), mode=["bilinear", "nearest"], keep_size=True),
            CenterCropByPercentd(keys=["img", "label"], percent=(0.25, 0.25)),
            EnsureTyped(keys=["img", "label"]),
            SaveImaged(keys=["img", "label"], output_dir="./zoom_cells", output_ext=".png", output_postfix="zoomed", allow_missing_keys=True, resample=False)
        ]
    )

    for i, j in zip(images, labels):
        zoom_transforms({"img": i, "label": j})


def reid(files, start_number=2000):
    n = start_number
    dir_name = "./gen/zoomed_cells"
    files.sort()
    for f in files:
        basename = os.path.basename(f)
        idx = "%05d" % n
        name = f"images/cell_{idx}.png" if 'label' not in basename else f"labels/cell_{idx}_label.png"
        os.rename(f, os.path.join(dir_name, name))
        if "label" in name:
            n += 1


if __name__ == '__main__':
    imgs = glob.glob("../../../data/Train_Pre_3class/images/*")
    labels = glob.glob("../../../data/Train_Pre_3class/labels/*")
    imgs.sort(), labels.sort()
    augment_small_cells(imgs[:141], labels[:141])

    files = glob.glob("./zoom_cells/**/*.png", recursive=True)
    reid(files)
