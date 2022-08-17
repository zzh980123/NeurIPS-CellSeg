# !/bin/bash -e
#python predict.py -i "/workspace/inputs/"  -o "/workspace/outputs/" --model_path ./work_dir/swinunetr_dfc_v3_fined/swinunetr_dfc_v3_3class_fined --model_name swinunetr_dfc_v3
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


############################### sdf predict #######################################

python predict_sdf_ls.py -i /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/  -o ./dev_workspace/swinunetr_sdf/outputs/ --model_path ./work_dir/swinunetr_sdf/swinunetr_sdf --model_name swinunetr --show_overlay
