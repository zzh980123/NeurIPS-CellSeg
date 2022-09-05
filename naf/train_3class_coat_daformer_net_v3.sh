#cd .. && pwd && python ./naf/model_training_3class_half_p16.py --data_path ./data/Train_Pre_3class_aug1 --work_dir ./naf/work_dir/coat_daformer_3class_s256_amp --model_name coat_daformer_net --input_size=256 --batch_size 8
cd .. && pwd && python ./naf/model_training_3class.py --data_path ./data/Train_Pre_3class_aug1 --work_dir ./naf/work_dir/coat_daformer_v3_3class_s512_amp --model_name coat_daformer_net_v3 --input_size=512 --batch_size 4 --initial_lr 6e-5
#cd .. && pwd && python ./naf/model_training_3class_half_p16.py --data_path ./data/Train_Pre_3class_aug1 --work_dir ./naf/work_dir/coat_daformer_3class_s768_amp --model_name coat_daformer_net --input_size=768 --batch_size 3 --num_workers 2 --initial_lr 6e-3
#cd .. && pwd && python ./naf/model_training_3class.py --data_path ./data/Train_Pre_3class_aug1 --work_dir ./naf/work_dir/coat_daformer_3class_s768 --model_name coat_daformer_net --input_size=768 --batch_size 2 --num_workers 2
