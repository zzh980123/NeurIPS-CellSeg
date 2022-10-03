#cd .. && pwd && python ./naf/model_training_grad.py --data_path ./data/Train_Pre_grad_aug1 --work_dir ./naf/work_dir/coat_daformer_grad_s512 --batch_size 7 --grad_lambda 1
#cd .. && pwd && python ./naf/model_training_grad.py --data_path ./data/Train_Pre_grad_aug1 --work_dir ./naf/work_dir/coat_daformer_grad2_s512 --batch_size 4 --grad_lambda 0.1 --continue_train True
#cd .. && pwd && python ./naf/model_training_grad.py --data_path ./data/Train_Pre_grad_aug1 --work_dir ./naf/work_dir/coat_daformer_grad_s512 --batch_size 7 --grad_lambda 0.1
#cd .. && pwd && python ./naf/model_training_grad.py --data_path ./data/Train_Pre_grad_aug1 --work_dir ./naf/work_dir/coat_daformer_grad_s512 --batch_size 7 --grad_lambda 10
cd .. && pwd && python ./naf/model_training_grad_plus.py --data_path ./data/Train_Pre_grad_aug1 --work_dir ./naf/work_dir/coat_daformer_grad_v3_plus_s512 --batch_size 4 --grad_lambda 0.2
