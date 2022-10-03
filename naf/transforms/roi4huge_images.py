import glob
import os
import random

from monai.data import PILReader, PILWriter

import transforms.utils as tra
from monai.transforms import (
    LoadImaged,
    AddChanneld,
    AsChannelFirstd,
    RandAffined,
    Compose,
    Invertd,
    RandSpatialCropd,
    SaveImaged, LoadImage, allow_missing_keys_mode, SaveImage,

    LoadImageD, AsChannelFirst, RandSpatialCrop, SaveImageD, ScaleIntensityRangePercentiles, NormalizeIntensity
)

random.seed(2022)

# random_roi_transformers = Compose(
#     [
#         LoadImaged(keys=["img"], reader=PILReader),
#         AsChannelFirstd(keys=["img"], allow_missing_keys=True),
#         RandSpatialCropd(['img'], max_roi_size=(1024, 1024)),
#         # RandAffined(keys=["img"], prob=1, rotate_range=(-3.14, 3.14)),
#         SaveImaged(keys=["img"], output_dir="./results", output_ext=".png", output_postfix="roi", writer=PILWriter)
#     ]
# )


import tifffile
import tqdm
if __name__ == '__main__':
    path = '/media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/Train-Unlabeled/release-part2-whole-slide'
    image_paths = glob.glob(os.path.join(path, "*"))
    sample_num = 50

    cf = AsChannelFirst()
    rsc = RandSpatialCrop(roi_size=(512, 512), max_roi_size=(1024, 1024))
    # nor = NormalizeIntensity()
    for image_path in tqdm.tqdm(image_paths, total=len(image_paths)):
        name = os.path.basename(image_path).split(".")[0]
        image = tifffile.imread(image_path)
        for i in tqdm.trange(sample_num):
            roi_image = rsc(cf(image)) / 65535 * 255
            # print(roi_image.max(), roi_image.min())
            save_ = SaveImage(output_dir=f"./results/{name}", output_ext=".png", output_postfix=f"roi_{i}", writer=PILWriter,separate_folder=False)
            save_(roi_image)

    image_paths = glob.glob(f"./results/**/*roi*.png", recursive=True)

    start_idx = 1713

    import shutil
    for i, image_path in tqdm.tqdm(enumerate(image_paths), total=len(image_paths)):
        idx = "%05d" % (start_idx + i)
        shutil.copy(image_path, f"../../data/Train_Pre_Unlabeled/unlabeled_cell_{idx}.png")




