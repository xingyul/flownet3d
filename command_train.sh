


python train.py \
    --model model_concat_upsa \
    --data data_preprocessing/data_processed_maxcut_35_20k_2k_8192 \
    --log_dir log_train \
    --num_point 2048 \
    --max_epoch 151 \
    --learning_rate 0.001 \
    --batch_size 16 \
    > log_train.txt 2>&1 &
