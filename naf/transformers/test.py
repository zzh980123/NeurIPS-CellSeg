from monai.data import PILReader, PILWriter

import transformers.utils as tra
from monai.transforms import (
    LoadImaged,
    AddChanneld,
    AsChannelFirstd,
    RandAffined,
    Compose,
    SaveImaged
)


test_bright = tra.RandBrightnessd(keys=["img"], allow_missing_keys=True, prob=1, add=(-50, 50))
test_hue = tra.RandHueAndSaturationd(keys=["img"], allow_missing_keys=True, prob=1, add=(-100, 100))
test_inverse = tra.RandInversed(keys=["img"], prob=1)

if __name__ == '__main__':
    image_test = "./datas/cell_00028.png"

    affine_transformers = Compose(
        [
            LoadImaged(keys=["img"], reader=PILReader),
            AsChannelFirstd(keys=["img"], allow_missing_keys=True),
            RandAffined(keys=["img"], prob=1, rotate_range=(-3.14, 3.14)),
            SaveImaged(keys=["img"], output_dir="./results", output_ext=".png", output_postfix="affine", writer=PILWriter)
        ]
    )

    brightness_transformers = Compose(
        [
            LoadImaged(keys=["img"], reader=PILReader),
            AsChannelFirstd(keys=["img"], allow_missing_keys=True),
            test_bright,
            SaveImaged(keys=["img"], output_dir="./results", output_ext=".png", output_postfix="brightness", writer=PILWriter)

        ]
    )

    hue_transformers = Compose(
        [
            LoadImaged(keys=["img"], reader=PILReader),
            AsChannelFirstd(keys=["img"], allow_missing_keys=True),
            test_hue,
            SaveImaged(keys=["img"], output_dir="./results", output_ext=".png", output_postfix="hue_sat", writer=PILWriter)
        ]
    )

    inverse_transformers = Compose(
        [
            LoadImaged(keys=["img"], reader=PILReader),
            AsChannelFirstd(keys=["img"], allow_missing_keys=True),
            test_inverse,
            SaveImaged(keys=["img"], output_dir="./results", output_ext=".png", output_postfix="inverse", writer=PILWriter)
        ]
    )

    res0 = affine_transformers({"img": image_test})
    res1 = brightness_transformers({"img": image_test})
    res2 = hue_transformers({"img": image_test})
    res3 = inverse_transformers({"img": image_test})

