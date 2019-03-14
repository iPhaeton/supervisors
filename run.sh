python jobs.py \
    --job_name=siamese \
    --model_name=deep_sort_cnn \
    --model_path=../input/models/deep_sort_cnn/freeze_model.py \
    --source_path=../input/mars/bbox_train/ \
    --epochs=11 \
    --num_per_class=4 \
    --loss=triplet_all \
    --metric=cosine \
    --margin=0.5 \
    --lr=1e-5 \
    --log_every=1 \
    --save_dir=./checkpoints \
    --save_every=500 \
    --validate_every=100 \
    --normalized=1 \
    --num_classes=3 \
    --batch_size=3
