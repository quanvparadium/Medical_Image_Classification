python training_main/main_linear.py\
 --dataset 'Chest_MedFM' \
 --pretrained\
 --competition 1 --backbone "swin_L_384_22k" --with_transformer_head \
 --save_folder '/content/drive/MyDrive/save_weight/chest' \
 --num_class 19 --epochs 12 --save_freq 3 --print_freq 50 --batch_size 8 \
 --img_size 384 --hidden_dim 2048 \
 --keep_input_proj --dim_feedforward 8192 \
 --amp\
 --nshot 1 \
 --train_csv_path './data/Datasets/chest/chest_1-shot_train_exp1.txt'\
 --val_csv_path './data/Datasets/chest/chest_1-shot_val_exp1.txt' \
 --test_csv_path './chest_val.csv'\
 --train_image_path '/content/drive/MyDrive/Med_Grand_Challenge/MedFMC_train/chest/images'\
 --val_image_path '/content/drive/MyDrive/Med_Grand_Challenge/MedFMC_train/chest/images' \
 --test_image_path '/content/drive/MyDrive/Med_Grand_Challenge/MedFMC_val/chest/images'