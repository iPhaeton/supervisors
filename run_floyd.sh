python jobs.py \
    --job_name=siamese \
    --model_name=deep_sort_cnn \
    --source_path=../input/mars/bbox_train/ \
    --model_path=../input/models/deep_sort_cnn/freeze_model.py \
    --checkpoint_path=../input/models/deep_sort_cnn/mars-small128.ckpt-68577 \
    --batch_size=32 \
    --epochs=101 \
    --num_per_class=12 \
    --loss=triplet_semihard \
    --metric=cosine \
    --margin=1 \
    --lr=1e-5 \
    --log_every=1 \
    --save_dir=./checkpoints \
    --save_every=10 \
    --validate_every=1 \
    --normalized=0 \
    --normalized_input=0 \
    --log_dir="./log"

#--checkpoint_path=../input/models/checkpoint/iteration-90.ckpt \
