#cd .. && pwd && python ./naf/model_training_grad.py --data_path ./data/Train_Pre_grad_aug1 --work_dir ./naf/work_dir/coat_daformer_grad_s512 --batch_size 7 --grad_lambda 1
#cd .. && pwd && python ./naf/model_training_grad.py --data_path ./data/Train_Pre_grad_aug1 --work_dir ./naf/work_dir/coat_daformer_grad2_s512 --batch_size 4 --grad_lambda 0.1 --continue_train True
#cd .. && pwd && python ./naf/model_training_grad.py --data_path ./data/Train_Pre_grad_aug1 --work_dir ./naf/work_dir/coat_daformer_grad_s512 --batch_size 7 --grad_lambda 0.1
# cd .. && pwd && python ./naf/model_training_grad.py --data_path ./data/Train_Pre_grad_aug1 --work_dir ./naf/work_dir/coat_daformer_grad_s512 --batch_size 4 --grad_lambda 0.5 --model_name coat_daformer_net_grad_v3p1

# freeze encoder
#python ./naf/model_training_grad.py --data_path ./data/Train_Pre_grad_aug2 --work_dir ./naf/work_dir/coat_daformer_grad_s512_fzecd/ --batch_size 4 --grad_lambda 0.5 --model_name coat_daformer_net_grad_v3 --freeze_encoder
# unfreeze
#python ./naf/model_training_grad.py --data_path ./data/Train_Pre_grad_aug2 --work_dir ./naf/work_dir/coat_daformer_grad_s512_fzecd/ --batch_size 4 --grad_lambda 0.5 --model_name coat_daformer_net_grad_v3 --continue_train

# cd .. && pwd && python ./naf/model_training_grad.py --data_path ./data/Train_Pre_grad_aug2 --work_dir ./naf/work_dir/coat_daformer_grad_s512/ --batch_size 4 --grad_lambda 0.5 --model_name coat_daformer_net_grad_v3
# aux
#python ./naf/model_training_grad.py --data_path ./data/Train_Pre_grad_aug2 --work_dir ./naf/work_dir/coat_daformer_grad_s512/ --batch_size 4 --grad_lambda 0.5 --model_name coat_daformer_net_grad_v3_aux --continue_train

# v3_n
#python ./naf/model_training_grad.py --data_path ./data/Train_Pre_grad_aug2 --work_dir ./naf/work_dir/coat_daformer_grad_n_s512/ --batch_size 4 --grad_lambda 0.5 --model_name coat_daformer_net_grad_v3_n
# v3_involution only
#python ./naf/model_training_grad.py --data_path ./data/Train_Pre_grad_aug2 --work_dir ./naf/work_dir/coat_daformer_grad_v3_s512/ --batch_size 5 --grad_lambda 0.5 --model_name coat_daformer_net_grad_v3_invcon

#cd .. && pwd && python ./naf/model_training_grad.py --data_path ./data/Train_Pre_grad_aug1 --work_dir ./naf/work_dir/coat_daformer_grad_s512 --batch_size 7 --grad_lambda 10
#cd .. && pwd && python ./naf/model_training_grad_plus.py --data_path ./data/Train_Pre_grad_aug1 --work_dir ./naf/work_dir/coat_daformer_grad_v3_plus_s512 --batch_size 4 --grad_lambda 0.2
