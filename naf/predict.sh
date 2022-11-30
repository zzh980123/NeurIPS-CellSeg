# !/bin/bash -e
#python predict.py -i "/workspace/inputs/"  -o "/workspace/outputs/" --model_path .//best_model --model_name coat_daformer_net_grad_v3

#python ./naf/predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./naf/dev_workspace/coat_daformer_v3_3class_s512_fined/outputs_s768/ --model_path ./naf/work_dir/coat_daformer_v3_3class_s512/coat_daformer_net_v3_3class_fined --model_name coat_daformer_net_v3 --show_overlay --input_size 768
#python predict.py -i ../data/Train_Pre_Unlabeled  -o ./dev_workspace/coat_daformer_3class_s512_amp_3class/unlabeled/outputs_s768_fined/ --model_path ./work_dir/coat_daformer_3class_s512_fined/coat_daformer_net_3class_fined --model_name coat_daformer_net --show_overlay --input_size 768
#python predict.py -i ../data/Train_Pre_Unlabeled  -o ./dev_workspace/coat_daformer_3class_s512_amp_3class/unlabeled/outputs_s768_fined/ --model_path ./work_dir/coat_daformer_3class_s512_fined/coat_daformer_net_3class_fined --model_name coat_daformer_net --show_overlay --input_size 768
#

############################### sdf predict ######################################

#python predict_sdf_ls.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/swinunetr_sdf/outputs/ --model_path ./work_dir/swinunetr_sdf/swinunetr_sdf --model_name swinunetr --show_overlay

############################### grad predict #####################################
#grad3 use lavzloss but not use direction loss
#grad2 use all losses
#grad is the original one
# grad3
#python predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/coat_daformer_grad3_s512/outputs_s512_watershed/ --model_path ./work_dir/coat_daformer_grad3_s512/coat_daformer_net_grad_grad --model_name coat_daformer_net_grad --show_overlay --input_size 512
# grad2
#python predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/coat_daformer_grad2_s512/outputs_s768_watershed/ --model_path ./work_dir/coat_daformer_grad2_s512/coat_daformer_net_grad_grad --model_name coat_daformer_net_grad --show_overlay --input_size 768
# grad2_v2
#python predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/coat_daformer_grad2_v2_s512/outputs_s768_watershed/ --model_path ./work_dir/coat_daformer_grad2_s512/coat_daformer_net_grad_v2_grad --model_name coat_daformer_net_grad_v2 --show_overlay --input_size 768
# grad4
#python predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/coat_daformer_grad4_s512/outputs_s768_watershed/ --model_path ./work_dir/coat_daformer_grad4_s512/coat_daformer_net_grad_grad --model_name coat_daformer_net_grad --show_overlay --input_size 512
# grad_v3 / v3_2
#python naf/predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./naf/dev_workspace/coat_daformer_grad_v3_s512/outputs_s768/ --model_path ./naf/work_dir/coat_daformer_grad_v3_s512/coat_daformer_net_grad_v3_grad --model_name coat_daformer_net_grad_v3 --show_overlay --input_size 768
# python ./naf/predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./naf/dev_workspace/coat_daformer_grad_v3_2_s512/outputs_s768_watershed/ --model_path ./naf/work_dir/coat_daformer_grad_v3_2_s512/coat_daformer_net_grad_v3_grad --model_name coat_daformer_net_grad_v3 --show_overlay --input_size 768 --watershed
#python naf/predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./naf/dev_workspace/coat_daformer_grad_v3_2_s512/outputs_s768_watershed/ --model_path ./naf/work_dir/coat_daformer_grad_v3_2_s512/coat_daformer_net_grad_v3_grad --model_name coat_daformer_net_grad_v3 --show_overlay --input_size 768 --watershed
#python naf/predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./naf/dev_workspace/coat_daformer_grad_v3_3_s512/outputs_s768_watershed/ --model_path ./naf/work_dir/coat_daformer_grad_v3_3_s512/coat_daformer_net_grad_v3_grad --model_name coat_daformer_net_grad_v3 --show_overlay --input_size 768 --watershed
#python predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/coat_daformer_grad_v3_2_s512/outputs_s768/ --model_path ./work_dir/coat_daformer_grad_v3_2_s512/coat_daformer_net_grad_v3_grad --model_name coat_daformer_net_grad_v3 --show_overlay --input_size 768
# cd .. && python naf/predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o .naf/dev_workspace/coat_daformer_grad_v3_2_s512/outputs_s768/ --model_path ./naf/work_dir/coat_daformer_grad_v3_2_s512/coat_daformer_net_grad_v3_grad --model_name coat_daformer_net_grad_v3 --show_overlay --input_size 768
# grad_v3_aux
# python ./naf/model_training_grad.py --data_path ./data/Train_Pre_grad_aug2 --work_dir ./naf/work_dir/coat_daformer_grad_s512/ --batch_size 4 --grad_lambda 0.5 --model_name coat_daformer_net_grad_v3_aux --continue_train
#python naf/predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./naf/dev_workspace/coat_daformer_grad_v3_aux_s512/outputs_s768/ --model_path ./naf/work_dir/coat_daformer_grad_s512/coat_daformer_net_grad_v3_aux_grad --model_name coat_daformer_net_grad_v3_aux --show_overlay --input_size 768

## SWA ##
# this test use epoch 188 0.8466 version
#python predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/coat_daformer_grad3_swa_s512/outputs_s768/ --model_path ./work_dir/coat_daformer_grad_swa_s512/coat_daformer_net_grad_grad --model_name coat_daformer_net_grad --show_overlay --input_size 768
# this test
# wonderwork plan todo doker...
#python naf/predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./naf/dev_workspace/coat_daformer_grad_v3_swa_s512/outputs_s768_watershed/ --model_path ./naf/work_dir/coat_daformer_grad_swa_s512/coat_daformer_net_grad_v3_grad --model_name coat_daformer_net_grad_v3 --show_overlay --input_size 768 --watershed

## mean teacher ##
#python naf/predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./naf/dev_workspace/coat_daformer_grad_v3_4_mean_teacher_s512/outputs_s768/ --model_path ./naf/work_dir/coat_daformer_grad_v3_4_s512/coat_daformer_net_grad_v3_mean_teacher --model_name coat_daformer_net_grad_v3 --show_overlay --input_size 768
# cd .. && python naf/predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./naf/dev_workspace/coat_daformer_grad_v3_2_mean_teacher_s512/outputs_s512/ --model_path ./naf/work_dir/coat_daformer_grad_v3_2_s512/coat_daformer_net_grad_v3_mean_teacher --model_name coat_daformer_net_grad_v3 --show_overlay --input_size 512 --watershed
# python naf/predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./naf/dev_workspace/coat_daformer_grad_v3_mean_teacher_s512/outputs_s768/ --model_path ./naf/work_dir/coat_daformer_grad_v3_s512/coat_daformer_net_grad_v3_mean_teacher --model_name coat_daformer_net_grad_v3 --show_overlay --input_size 768

## center ##
#cd .. && python naf/predict_center.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./naf/dev_workspace/coat_daformer_center_s512/outputs_s768/ --model_path ./naf/work_dir/coat_daformer_center_s512/coat_daformer_net_center_grad --model_name coat_daformer_net_center --show_overlay --input_size 768


## unseen cells predict ##
python naf/predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/test_cells/MSC计数  -o ./naf/dev_workspace/test_cells/MSC/coat_daformer_grad_v3_2_s512/outputs_s512_watershed/ --model_path ./naf/work_dir/coat_daformer_grad_v3_s512/backup --model_name coat_daformer_net_grad_v3 --show_overlay --input_size 512 --watershed
# python naf/predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/test_cells/293T  -o ./naf/dev_workspace/test_cells/293T/coat_daformer_grad_v3_2_s512/outputs_s512_watershed/ --model_path ./naf/work_dir/coat_daformer_grad_v3_2_s512/coat_daformer_net_grad_v3_grad --model_name coat_daformer_net_grad_v3 --show_overlay --input_size 512 --watershed --tta
# python naf/predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/test_cells/HePG2  -o ./naf/dev_workspace/test_cells/HePG2/coat_daformer_grad_v3_2_s512/outputs_s512_watershed/ --model_path ./naf/work_dir/coat_daformer_grad_v3_2_s512/coat_daformer_net_grad_v3_grad --model_name coat_daformer_net_grad_v3 --show_overlay --input_size 512 --watershed --tta


## #python ./naf/model_training_grad.py --data_path ./data/Train_Pre_grad_aug2 --work_dir ./naf/work_dir/coat_daformer_grad_v3_s512/ --batch_size 5 --grad_lambda 0.5 --model_name coat_daformer_net_grad_v3_invcon
#python naf/predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./naf/dev_workspace/coat_daformer_grad_v3_invcon_s512/outputs_s768_watershed/ --model_path ./naf/work_dir/coat_daformer_grad_v3_s512/coat_daformer_net_grad_v3_invcon_grad --model_name coat_daformer_net_grad_v3_invcon --show_overlay --input_size 768 --watershed

## backup
#python naf/predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./naf/dev_workspace/coat_daformer_grad_v3_s512/outputs_s768_watershed_tta/ --model_path .--model_path/naf/work_dir/coat_daformer_grad_v3_s512/backup --model_name coat_daformer_net_grad_v3 --show_overlay --input_size 768 --watershed --tta
#python naf/predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./naf/dev_workspace/coat_daformer_grad_v3_s512/outputs_s768/ --model_path ./naf/work_dir/coat_daformer_grad_v3_s512/backup --model_name coat_daformer_net_grad_v3 --show_overlay --input_size 768
