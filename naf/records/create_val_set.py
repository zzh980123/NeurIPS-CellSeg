
import pickle, shutil, os.path

if __name__ == '__main__':
    with open('naf/records/val_dataset.pt', 'rb') as f:
        val_files = pickle.load(f)

    os.makedirs('naf/records/val/images')
    os.makedirs('naf/records/val/labels')

    for a in val_files:
        print(a)
        img_path = a['img']
        label_path = a['label']

        shutil.copy(img_path, 'naf/records/val/images')
        shutil.copy(label_path, 'naf/records/val/labels')

