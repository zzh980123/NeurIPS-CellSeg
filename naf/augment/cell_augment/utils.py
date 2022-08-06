import glob
import os

import numpy as np
import imageio as io
# taichi's GUI will interpolate position for every particle at the pixels (the position is not always at the middle of the pixel)... we need color the data by ourself.
from numba import jit


def simulation_data2image(path, image_path, size=(256, 256)):
    data = np.load(path)
    datas = data['pos'], data['items']
    data.close()

    res = np.zeros(size)
    res = simulation_data2image_inner(datas, res, size)

    io.imwrite(image_path, res.astype(np.uint8))


@jit(nopython=True)
def simulation_data2image_inner(data, output, size):
    positions = data[0]
    item = data[1]

    particle_num = positions.shape[0]
    it_num = item.shape[0]
    assert it_num == particle_num

    for i in range(particle_num):
        xy = positions[i]
        item_idx = item[i]

        h = xy[0] * size[0]
        w = xy[1] * size[1]

        output[int(h), int(w)] = int(item_idx)

    return output


def convert_sim_data2png(data_dir, save_dir, size=(256, 256)):
    files = glob.glob(f"{data_dir}/*.npz")
    for f in files:
        fn = os.path.basename(f)
        save_path = os.path.join(save_dir, fn.replace(".npz", "_label.png"))
        simulation_data2image(f, save_path, size)


if __name__ == '__main__':
    convert_sim_data2png("simulation_data", "gen_labels")
