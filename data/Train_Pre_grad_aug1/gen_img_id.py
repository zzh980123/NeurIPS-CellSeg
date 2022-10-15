import glob
import os

if __name__ == '__main__':
    images = glob.glob("./images/*")
    images.sort()
    names = {}
    """
    0: Bright Filed
    1: DIC
    2: PC
    3: Fluorescent 
    """


    def name2id(name: str):
        return int(name.split("_")[1].split(".")[0])


    def rules(idx):
        if 1 <= idx <= 300 or 1000 <= idx < 1126 or idx >= 2000:
            return 0
        elif 301 <= idx <= 500:
            return 1
        elif 501 <= idx <= 700 or 1001 <= idx <= 1073:
            return 2
        elif 701 <= idx <= 1000 or 1126 <= idx <= 1137 or 1074 <= idx <= 1113:
            return 3


    for img in images:
        cell_idx = name2id(img)
        class_id = rules(cell_idx)
        names[cell_idx] = class_id

    import json

    with open("class_map.json", "w") as f:
        json.dump(names, f)
