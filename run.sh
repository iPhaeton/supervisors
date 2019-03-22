batch_size=32
num_per_class=4
loss=triplet_hard
metric=eucledian_squared
normalized_input=2
lr=1e-5

python jobs.py \
    --job_name=siamese \
    --model_name=deep_sort_cnn \
    --source_path=../input/mars/bbox_train/ \
    --model_path=../input/models/deep_sort_cnn/freeze_model.py \
    --batch_size=$batch_size \
    --epochs=10 \
    --num_per_class=4 \
    --loss=$loss \
    --metric=$metric \
    --margin=1 \
    --lr=$lr \
    --log_every=1 \
    --save_dir=./checkpoints \
    --save_every=1000 \
    --validate_every=1 \
    --normalized=0 \
    --normalized_input=$normalized_input \
    --log_dir="./logs/log-${batch_size}-${num_per_class}-${loss}-${metric}-${normalized_input}-${lr}"

#--checkpoint_path=../input/models/deep_sort_cnn/mars-small128.ckpt-68577 \