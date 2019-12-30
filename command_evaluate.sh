


python evaluate.py \
    --model model_concat_upsa \
    --dataset flying_things_dataset \
    --data data_preprocessing/data_processed_maxcut_35_20k_2k_8192 \
    --log_dir log_evaluate \
    --model_path log_train/model.ckpt \
    --num_point 2048 \
    --batch_size 16 \
    > log_evaluate.txt 2>&1 &
