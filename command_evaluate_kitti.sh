


python evaluate.py \
    --model model_concat_upsa_eval_kitti \
    --gpu 1 \
    --dataset kitti_dataset \
    --data kitti_rm_ground \
    --log_dir log_evaluate \
    --model_path log_train/model.ckpt \
    --num_point 16384 \
    --batch_size 1 \
    > log_evaluate.txt 2>&1 &
