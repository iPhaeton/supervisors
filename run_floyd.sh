python jobs.py \
    --job_name=siamese \
    --model_name=deep_sort_cnn \
    --source_path=../input/mars/bbox_train/ \
    --model_path=../input/models/deep_sort_cnn/freeze_model.py \
    --batch_size=32 \
    --epochs=100 \
    --num_per_class=12 \
    --loss=triplet_all \
    --metric=eucledian_squared \
    --margin=1 \
    --lr=1e-3 \
    --log_every=1 \
    --save_dir=./checkpoints \
    --save_every=10 \
    --validate_every=1 \
    --normalized=0 \
    --log_dir="./log"
