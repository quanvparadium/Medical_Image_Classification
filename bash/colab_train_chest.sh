python training_main/main_linear.py\
 --dataset 'Chest_MedFM' \
 --pretrained\
 --competition 1 --backbone "swin_B_384_22k" --with_transformer_head \
 --save_folder './save_linear/chest' \
 --num_class 19 --epochs 20 --save_freq 5 --print_freq 50 --batch_size 8 \
 --img_size 384 --hidden_dim 2048 \
 --keep_input_proj --dim_feedforward 8192 \
 --amp\
 --train_csv_path './data/Datasets/chest/chest_1-shot_train_exp1.csv'\
 --test_csv_path './chest_val.csv'\
 --train_image_path '/content/drive/MyDrive/Med_Grand_Challenge/MedFMC_train/chest/images'\
 --test_image_path './contenet/drive/MyDrive/Med_Grand_Challenge/MedFMC_val/chest/images'