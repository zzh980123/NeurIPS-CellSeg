import tifffile as tif
import imageio
import numpy as np

# path = '/media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/Train-Labeled/labels/cell_00143_label.tiff'

def view_label_3class(path, path_src):
    path = '../../data/Train_Pre_3class/labels/cell_00038_label.png'
    path_src = '../../data/Train_Pre_3class/images/cell_00038.png'
    # label = tif.imread(path)
    label = imageio.imread_v2(path)
    src = imageio.imread_v2(path_src)
    print(label)
    label = np.array(label)
    print(label.shape)
    print(np.unique(label))

    # label_val = list(np.unique(label))
    #
    # shape = label.shape
    #
    # label_color = np.zeros(shape + (3, ))
    # rgb_list = list()
    # for lv in label_val:
    #     Blue = lv & 255
    #     Green = (lv >> 8) & 255
    #     Red = (lv >> 16) & 255
    #     rgb_list.append([Red, Green, Blue])
    # #
    # # for c in range(3):
    # #     label_color[..., c] = label / label.max() * (c * 256 / 3)
    # label_color[..., 2] = label / label.max() * (3 * 256 / 3)
    # imageio.imwrite("view.png", label_color)

    label[label == 0] = 0
    label[label == 1] = 255
    # label[label == 1] = 255
    imageio.imwrite("view.png", label)
    imageio.imwrite("view_src.png", src)


def view_label_sdf(path):
    path = 'cell_00001_label.png'
    label = imageio.imread_v2(path)
    print(np.unique(label))
    outline_ = label <= 127
    label[outline_] = 255
    label[np.logical_not(outline_)] = 0

    imageio.imwrite("outline.png", label)


if __name__ == '__main__':
    # view_label_sdf("")
    view_label_3class("", "")
