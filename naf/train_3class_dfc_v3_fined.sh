cd .. && pwd && python ./naf/model_training_3class_fined.py --data_path ./data/Train_Pre_3class --work_dir ./naf/work_dir/swinunetr_dfc_v3_fined --model_name swinunetr_dfc_v3 --initial_lr 6e-3 --model_path ./naf/work_dir/swinunetr_dfc_v3/swinunetr_dfc_v3_3class
#cd .. && pwd && python ./naf/model_training_3class.py --data_path ./data/Train_Pre_3class --work_dir ./naf/work_dir/swinunetr_dfc_v3_2 --model_name swinunetr_dfc_v3 --initial_lr 6e-4