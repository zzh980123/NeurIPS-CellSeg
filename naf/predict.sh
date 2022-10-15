# !/bin/bash -e
#python predict.py -i "/workspace/inputs/"  -o "/workspace/outputs/" --model_path ./work_dir/best_model --model_name coat_daformer_net
#python predict.py -i "/media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/"  -o "./dev_workspace/outputs/" --model_path "./work_dir/swinunetrv2_3class" --show_overlay
#python predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/view11_elastic/outputs/ --model_path ./work_dir/view11_elastic/swinunetrv2_3class --show_overlay
#python predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/v3/outputs/ --model_path ./work_dir/v3/swinunetrv3_3class --model_name swinunetrv3 --show_overlay
#python predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/swinunetrv2_dfc/outputs/ --model_path ./work_dir/swinunetrv2_dfc/swinunetrv2_3class --model_name swinunetrv2 --show_overlay
#python predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/swinunetr_dfc_v3/outputs/ --model_path ./work_dir/swinunetr_dfc_v3/swinunetr_dfc_v3_3class --model_name swinunetr_dfc_v3 --show_overlay
#python predict_global_info.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/baseline_global_info/outputs/ --model_path ./work_dir/baseline_global_info/swinunetr_3class --model_name swinunetr --show_overlay
#python predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/swinunetrstyle_3class/outputs/ --model_path ./work_dir/swinunetrstyle_3class/swinunetrstyle_3class --model_name swinunetrstyle --show_overlay
#python predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/swinunetr_dfc_v3_3class_fined_post/outputs/ --model_path ./work_dir/swinunetr_dfc_v3_fined/swinunetr_dfc_v3_3class_fined --model_name swinunetr_dfc_v3 --show_overlay
#python predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/swinunetr_dfc_v4_3class/outputs/ --model_path ./work_dir/swinunetr_dfc_v4/swinunetr_dfc_v4_3class --model_name swinunetr_dfc_v4 --show_overlay
#python predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/swinunetrstyle_aug_3class/outputs/ --model_path ./work_dir/swinunetrstyle_aug_3class/swinunetrstyle_3class --model_name swinunetrstyle --show_overlay
#python predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/swinunetr_dfc_v4_fined_post/outputs/ --model_path ./work_dir/swinunetr_dfc_v4_fined/swinunetr_dfc_v4_3class_fined --model_name swinunetr_dfc_v4 --show_overlay
#python predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/swinunetrstyle_aug1_3class/outputs/ --model_path ./work_dir/swinunetrstyle_aug1_3class/swinunetrstyle_3class --model_name swinunetrstyle --show_overlay
#python predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/swinunetr_baselines/small/outputs/ --model_path ./work_dir/swinunetr_baselines_3class/small/swinunetr_emb48_2262_3class --model_name swinunetr_emb48_2262 --show_overlay
#python predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/swinunetr_baselines_fined/small/outputs/ --model_path ./work_dir/swinunetr_baselines_3class_fined/small/swinunetr_emb48_2262_3class_fined --model_name swinunetr_emb48_2262 --show_overlay
#python predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/swinunetr_baselines_fined/tiny/outputs/ --model_path ./work_dir/swinunetr_3class_fined/swinunetr_3class_fined --model_name swinunetr --show_overlay
#python predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/swinunetr_baselines_fined/tiny/outputs/ --model_path ./work_dir/swinunetr_3class_fined/swinunetr_3class_fined --model_name swinunetr --show_overlay
#python predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/swinunetr_baselines_fined_aug1/small/outputs/ --model_path ./work_dir/swinunetr_baselines_3class_fined_aug1/small/swinunetr_emb48_2262_3class_fined --model_name swinunetr_emb48_2262 --show_overlay
#python predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/swinunetr_dfc_v5/tiny2/outputs/ --model_path ./work_dir/swinunetr_dfc_v5/swinunetr_dfc_v5_3class --model_name swinunetr_dfc_v5 --show_overlay
#python predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/swinunetr_dfc_v4_fined/small/outputs/ --model_path ./work_dir/swinunetr_dfc_v4_fined/swinunetr_dfc_v4_emb48_2262_3class_fined --model_name swinunetr_dfc_v4_emb48_2262 --show_overlay
#python predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/swinunetr_dfc_v5_fined/tiny2/outputs/ --model_path ./work_dir/swinunetr_dfc_v5/swinunetr_dfc_v5_3class_fined --model_name swinunetr_dfc_v5 --show_overlay
#python predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/coat_daformer_3class_s512_amp_3class/outputs_s640/ --model_path ./work_dir/coat_daformer_3class_s512_amp/coat_daformer_net_3class --model_name coat_daformer_net --show_overlay --input_size 640
#python predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/coat_daformer_3class_s512_amp_3class/outputs_s768_o50_watershed_fined/ --model_path ./work_dir/coat_daformer_3class_s512_fined/coat_daformer_net_3class_fined --model_name coat_daformer_net --show_overlay --input_size 768
#python predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/coat_daformer_3class_s512_amp_3class/outputs_s768_o50_fined_bug_fixed/ --model_path ./work_dir/coat_daformer_3class_s512_fined/coat_daformer_net_3class_fined --model_name coat_daformer_net --show_overlay --input_size 768
#python predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/coat_daformer_3class_s768_3class/outputs_s768/ --model_path ./work_dir/coat_daformer_3class_s768/coat_daformer_net_3class --model_name coat_daformer_net --show_overlay --input_size 768
#python predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/coat_daformer_v2_s512_amp_3class/outputs_s768/ --model_path ./work_dir/coat_daformer_v2_3class_s512_amp/coat_daformer_net_v2_3class --model_name coat_daformer_net_v2 --show_overlay --input_size 768
#python predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/coat_daformer_v2_3class_s512_fined/outputs_s768_half/ --model_path ./work_dir/coat_daformer_v2_3class_s512_amp_fined/coat_daformer_net_v2_3class_fined --model_name coat_daformer_net_v2 --show_overlay --input_size 768
#python predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/coat_daformer_v3_3class_s512_fined/outputs_s768/ --model_path ./work_dir/coat_daformer_v3_3class_s512/coat_daformer_net_v3_3class_fined --model_name coat_daformer_net_v3 --show_overlay --input_size 768
#python predict.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/coat_daformer_v3_3class_s512_fined/outputs_s768/ --model_path ./work_dir/coat_daformer_v3_3class_s512/coat_daformer_net_v3_3class_fined --model_name coat_daformer_net_v3 --show_overlay --input_size 768
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
#python predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/coat_daformer_grad_v3_s512/outputs_s512_watershed/ --model_path ./work_dir/coat_daformer_grad_v3_s512/coat_daformer_net_grad_v3_grad --model_name coat_daformer_net_grad_v3 --show_overlay --input_size 512
#python predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/coat_daformer_grad_v3_s512/outputs_s768_watershed/ --model_path ./work_dir/coat_daformer_grad_v3_s512/coat_daformer_net_grad_v3_grad --model_name coat_daformer_net_grad_v3 --show_overlay --input_size 768
#python predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/coat_daformer_grad_v3_2_s512/outputs_s768_watershed/ --model_path ./work_dir/coat_daformer_grad_v3_2_s512/coat_daformer_net_grad_v3_grad --model_name coat_daformer_net_grad_v3 --show_overlay --input_size 768 --watershed
#python predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/coat_daformer_grad_v3_2_s512/outputs_s768/ --model_path ./work_dir/coat_daformer_grad_v3_2_s512/coat_daformer_net_grad_v3_grad --model_name coat_daformer_net_grad_v3 --show_overlay --input_size 768
#python predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/coat_daformer_grad_v3_2_s512/outputs_s768/ --model_path ./work_dir/coat_daformer_grad_v3_2_s512/coat_daformer_net_grad_v3_grad --model_name coat_daformer_net_grad_v3 --show_overlay --input_size 768


## SWA ##
# this test use epoch 188 0.8466 version
#python predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/coat_daformer_grad3_swa_s512/outputs_s768/ --model_path ./work_dir/coat_daformer_grad_swa_s512/coat_daformer_net_grad_grad --model_name coat_daformer_net_grad --show_overlay --input_size 768
# this test
#python predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/coat_daformer_grad3_swa_s512/outputs_s768_watershed/ --model_path ./work_dir/coat_daformer_grad_swa_s512/coat_daformer_net_grad_grad --model_name coat_daformer_net_grad --show_overlay --input_size 768

## mean teacher ##
cd .. && python naf/predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./naf/dev_workspace/coat_daformer_grad_v3_2_mean_teacher_s512/outputs_s768/ --model_path ./naf/work_dir/coat_daformer_grad_v3_2_s512/coat_daformer_net_grad_v3_mean_teacher --model_name coat_daformer_net_grad_v3 --show_overlay --input_size 768 --watershed
#cd .. && python naf/predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./naf/dev_workspace/coat_daformer_grad_v3_2_mean_teacher_s512/outputs_s512/ --model_path ./naf/work_dir/coat_daformer_grad_v3_2_s512/coat_daformer_net_grad_v3_mean_teacher --model_name coat_daformer_net_grad_v3 --show_overlay --input_size 512 --watershed
#cd .. && python naf/predict_grad.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./naf/dev_workspace/coat_daformer_grad_v3_mean_teacher_s512/outputs_s768/ --model_path ./naf/work_dir/coat_daformer_grad_v3_s512/coat_daformer_net_grad_v3_mean_teacher --model_name coat_daformer_net_grad_v3 --show_overlay --input_size 768 --watershed

## center ##
#cd .. && python naf/predict_center.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./naf/dev_workspace/coat_daformer_center_s512/outputs_s768/ --model_path ./naf/work_dir/coat_daformer_center_s512/coat_daformer_net_center_grad --model_name coat_daformer_net_center --show_overlay --input_size 768
