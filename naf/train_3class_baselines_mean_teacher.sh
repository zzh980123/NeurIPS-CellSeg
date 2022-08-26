#cd .. && pwd && python ./naf/model_training_3class.py --data_path ./data/Train_Pre_3class --work_dir ./naf/work_dir/swinunetr_baselines_3class/small --model_name swinunetr_emb48_2262
#cd .. && pwd && python ./naf/model_training_3class.py --data_path ./data/Train_Pre_3class --work_dir ./naf/work_dir/swinunetrv2_3class --model_name swinunetrv2
#cd .. && pwd && python ./naf/model_training_3class.py --data_path ./data/Train_Pre_3class --work_dir ./naf/work_dir/swinunetrv2_3class --model_name swinunetrv2
#cd .. && pwd && python ./naf/model_training_3class.py --data_path ./data/Train_Pre_3class --work_dir ./naf/work_dir/swinunetrv2_3class --model_name swinunetrv2
cd .. && pwd && python ./naf/model_training_3class_mean_teacher.py --labeled_path ./data/Train_Pre_3class_aug1 --unlabeled_path ./data/Train_Pre_Unlabeled  --work_dir ./naf/work_dir/swinunetr_baselines_3class/tiny2 --model_name swinunetr_emb24_2262
