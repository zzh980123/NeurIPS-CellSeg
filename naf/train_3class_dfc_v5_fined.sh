#cd .. && pwd && python ./naf/model_training_3class.py --data_path ./data/Train_Pre_3class --work_dir ./naf/work_dir/swinunetr_dfc_v4 --model_name swinunetr_dfc_v4 --initial_lr 6e-4
cd .. && pwd && python ./naf/model_training_3class_fined.py --data_path ./data/Train_Pre_3class_aug1 --work_dir ./naf/work_dir/swinunetr_dfc_v5 --model_name swinunetr_dfc_v5 --initial_lr 6e-4
