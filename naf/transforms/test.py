from monai.data import PILReader, PILWriter

import transforms.utils as tra
from monai.transforms import (
    LoadImaged,
    AddChanneld,
    AsChannelFirstd,
    RandAffined,
    Compose,
    Invertd,
    SaveImaged, LoadImage, allow_missing_keys_mode, SaveImage
)


test_bright = tra.RandBrightnessd(keys=["img"], allow_missing_keys=True, prob=1, add=(-50, 50))
test_hue = tra.RandHueAndSaturationd(keys=["img"], allow_missing_keys=True, prob=1, add=(-100, 100))
test_inverse = tra.RandInversed(keys=["img"], prob=1)
test_mixcut = tra.RandCutoutd(keys=['img'], prob=1)

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

    load = LoadImage(
        reader=PILReader
    )

    pre_transformers = Compose(
        [
            LoadImaged(keys=["img", "label"], reader=PILReader),
            AsChannelFirstd(keys=["img"], allow_missing_keys=True),
            # SaveImaged(keys=["img"], output_dir="./results", output_ext=".png", output_postfix="affine", writer=PILWriter)
        ]
    )

    # define post transforms
    post_transforms = Compose([
        RandAffined(keys=["img", "label"], prob=1, scale_range=(0.5, 0.51)),

        SaveImaged(keys=["img", "label"], output_dir="./results", output_ext=".png", output_postfix="forward_trans", writer=PILWriter),
    ])

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

    mixcut_transformers = Compose(
        [
            LoadImaged(keys=["img"], reader=PILReader),
            AsChannelFirstd(keys=["img"], allow_missing_keys=True),
            test_mixcut,
            SaveImaged(keys=["img"], output_dir="./results", output_ext=".png", output_postfix="mixcut", writer=PILWriter)
        ]
    )

    # image_ = {"img": image_test, "label": image_test}
    # age_trans = pre_transformers(image_)
    # image_trans = post_transforms(age_trans)
    #
    # with allow_missing_keys_mode(post_transforms):
    #     inverted_seg = post_transforms.inverse(image_trans)
    #     label = inverted_seg["img"]
    #     inverted_seg["label"] = label
    #
    # save = SaveImaged(keys=["label"], output_dir="./results", output_ext=".png", output_postfix="backward_trans", writer=PILWriter)
    # save(inverted_seg)

    res0 = affine_transformers({"img": image_test})
    res1 = brightness_transformers({"img": image_test})
    res2 = hue_transformers({"img": image_test})
    res3 = inverse_transformers({"img": image_test})
    res4 = mixcut_transformers({"img": image_test})

    # fixed_aug_inputs = mean_teacher_transforms(inputs)
    #
    # batch_data["label"] = teacher_model(fixed_aug_inputs).detach()
    # with allow_missing_keys_mode(mean_teacher_transforms):
    #     back_t_outputs = mean_teacher_transforms.inverse(batch_data)




