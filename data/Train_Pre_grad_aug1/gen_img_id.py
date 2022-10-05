import glob
import os

if __name__ == '__main__':
    images = glob.glob("./images/*")
    images.sort()
    names = {}
    """
    0: HE Bright Filed
    1: DCF
    2: yingguang
    3: xiangwei
    """


    def name2id(name: str):
        return int(name.split("_")[1].split(".")[0])


    def rules(idx):
        if 12 >= idx >= 0 or 15 <= idx <= 141 or 145 <= idx <= 230 or 2000 <= idx <= 2011 or idx >= 2014:
            return 0
        elif 301 <= idx <= 527 or 530 <= idx <= 546 or 559 <= idx <= 718 or 1114 <= idx < 1137:
            return 1
        elif 719 <= idx <= 1000:
            return 2
        else:
            return 3


    for img in images:
        cell_idx = name2id(img)
        class_id = rules(cell_idx)
        names[cell_idx] = class_id

    import json

    with open("class_map.json", "w") as f:
        json.dump(names, f)
