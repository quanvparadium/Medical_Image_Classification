python training_main/main_linear_competition.py\
 --dataset 'Competition' \
 --pretrained\
 --competition 1 --backbone "swin_B_384_22k" --with_transformer_head \
 --save_folder './save_linear' \
 --num_class 6 --epochs 30 --save_freq 2 --print_freq 50 --batch_size 16 \
 --img_size 384 --hidden_dim 2048 \
 --keep_input_proj --dim_feedforward 8192 \
 --amp\
 --train_csv_path './chest_train.csv'\
#  --val_csv_path '/mnt/HDD/Chau_Truong/SupCon_OCT_Clinical/final_competition_csv/Train_9_Val_1_csv/Biomarker_Data_for_val.csv'\
 --test_csv_path '/mnt/HDD/Chau_Truong/SupCon_OCT_Clinical/final_competition_csv/test_set_submission_template.csv'\
 --train_image_path '/mnt/HDD/Chau_Truong/data/Datasets'\
 --val_image_path '/mnt/HDD/Chau_Truong/data/Datasets'\
 --test_image_path '/mnt/HDD/Chau_Truong/IEEE_2023_Ophthalmic_Biomarker_Det/TEST'