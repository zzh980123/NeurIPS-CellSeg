# Watershed-enhanced cell instance segmentation based on guidance of gradient

This repository is the official implementation of [Watershed-enhanced cell instance segmentation based on guidance of gradient](TBA).

## Environments and Requirements

- Ubuntu 18.04
- CPU: intel 10900f; RAM: 128GB; GPU: 2080ti
- CUDA 10.2
- python 3.8
- pytorch 1.12.0

To install requirements:

```setup
pip install -r naf/requirements.txt
```

## Dataset

- [Challenge Dataset](https://neurips22-cellseg.grand-challenge.org/dataset/)
- TODO, [Synthesized data](TBA)
- Prepare the data with image folder and label folder: image folder saves the cell images, label image folder saves all cell instance/3-classes/flows/sdf labels
- If train the segmentation methods under mean-theacher framework, you need to prepare a Unlabeled image folder which contains the orginal cell images.

## Preprocessing

A brief description of preprocessing method
Running the data preprocessing code (different cell representations with different preprocssing methods):

### synthesized data

run naf/augment/cell_augment/taichi_example_modify.py to start the engine, generate the points-set files by click the window (manual) or mark the ```random_gen``` ```true``` (automatic, todo. the script not finished, pelease modify the variables in the file). Then run gen/utils.py to generate the images and instance labels. Finally, move the images and labels to the dataset or preprocess them to 3-classes or flows.

```python
circle_shape_reset() # create circle cells
stripe_shape_reset() # create stripe cells
```

You can modify the ```w_n``` and ```cell_num``` to change the number of cell each row and the number of cells in one image.

### 3 classes label

```python
python data/pre_process_3class.py --input_path <path_to_input_data> --output_path <path_to_output_data>
```

### flow label

flow label contain 5 channels, pelease see naf/transforms/test_flow_gen.py in detail

```python
python naf/transforms/test_flow_gen.py --input_path <path_to_input_data> --output_path <path_to_output_data>
```

### only sdf label

```python
python naf/sdf_gen.py --input_path <path_to_input_data> --output_path <path_to_output_data>
```

## Training

To train the model(s) in the paper, run this command:

```train
python ./naf/model_training_grad.py --data_path <data_path> --work_dir <model_save_path> --batch_size <batch_size> --grad_lambda <grad_loss_hyper_parameter> --model_name <model_name>
```

```train example
python ./naf/model_training_grad.py --data_path ./data/Train_Pre_grad_aug2 --work_dir ./naf/work_dir/coat_daformer_grad_s512/ --batch_size 4 --grad_lambda 0.5 --model_name coat_daformer_net_grad_v3
```

More examples are under the **naf** folder which format are *train_\*.sh* and mean teacher based methods can be find here.

## Trained Models

You can download trained models here:

- [My awesome model](https://pan.baidu.com/s/1oenPzlN6796AnNBjeQOWUQ?pwd=khak) trained on the above dataset with the above code.
- [COAT pretrained model](https://vcl.ucsd.edu/coat/pretrained/coat_lite_medium_384x384_f9129688.pth) please put the file into naf/models/PretrainedModel/CoAT.

## Inference

To infer the testing cases, run this command:

```python
python inference.py --input_data <path_to_data> --label_path <label_path> --model_name <model_name> --model_path <path_to_trained_model> --output_path <path_to_output_data> 
```

example with overlay and grad images

```python
naf/test_grad.py -i naf/records/val/images -l naf/records/val/labels -o naf/records/results/coat_daformer_grad_v3_s512_o768_mt --model_path naf/work_dir/coat_daformer_grad_v3_s512/coat_daformer_net_grad_v3_mean_teacher --model_name coat_daformer_net_grad_v3 --show_overlay --show_grad --input_size 768
```

example with TTA

```python
naf/test_grad.py -i naf/records/val/images -l naf/records/val/labels -o naf/records/results/coat_daformer_grad_v3_s512_o768_mt --model_path naf/work_dir/coat_daformer_grad_v3_s512/coat_daformer_net_grad_v3_mean_teacher --model_name coat_daformer_net_grad_v3 --show_overlay --show_grad --tta --input_size 768
```

## Evaluation

To compute the evaluation metrics, run:

```eval
python baseline/compute_metric.py --seg_data <path_to_inference_results> --gt_data <path_to_ground_truth>
```

## Results

Our method achieves the following performance on [NeurIPS Cell Segmentation Challenge](https://neurips22-cellseg.grand-challenge.org/neurips22-cellseg/)

| Model name       |  F1 score  | test time |
| ---------------- | :----: | :--------------------: |
| Best Model | 77.24% |         -          |

## Contributing

This repository is released under the Apache License 2.0. License can be found in LICENSE file.

## Acknowledgement

> We thank the contributors of public datasets.
> Thanks to [Cellpose](https://github.com/MouseLand/cellpose) and [Tachi](https://github.com/taichi-dev/taichi).
