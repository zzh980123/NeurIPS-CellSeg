import glob

from skimage import measure

import transformers.flow_gen as flow
import tifffile
import numpy as np


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


import matplotlib.pyplot as plot
import tqdm
import imageio
if __name__ == '__main__':
    # label_root = '/media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/Train-Labeled/labels'
    # flow_root = '/media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/Train-Labeled/flows'

    is_3classes = True

    label_root = '../../data/Train_Pre_3class_aug1/labels'
    flow_root = '../../data/Train_Pre_3class_aug1/flows'

    files = glob.glob(label_root + "/*")
    flow_files = [f.replace(label_root, flow_root) for f in files]

    for f, ff in tqdm.tqdm(zip(files, flow_files), total=len(files)):
        if is_3classes:
            classes_3 = imageio.imread_v2(f)
            # to two classes
            classes_3[classes_3 > 1] = 0
            instance = measure.label(classes_3)
        else:
            instance = imageio.imread_v2(f)
        res = flow.labels_to_flows([instance], [ff], use_gpu=True, device="cuda:0")

    # for s in res:
    #     im = dx_to_circ(s[2:, :, :])
    #     print(im.shape)
    #
    #     plot.imshow(im)
    #     plot.show()
    #     plot.imshow(np.column_stack([s[0], s[1]]))
    #     plot.show()
