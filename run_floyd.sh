python jobs.py \
    --job_name=siamese \
    --model_name=deep_sort_cnn \
    --source_path=../input/mars/bbox_train/ \
    --model_path=../input/models/deep_sort_cnn/freeze_model.py \
    --checkpoint_path=../input/models/deep_sort_cnn/mars-small128.ckpt-68577 \
    --batch_size=100 \
    --epochs=500 \
    --num_per_class=4 \
    --loss=triplet_semihard \
    --metric=cosine \
    --margin=0.2 \
    --lr=1e-3 \
    --log_every=5 \
    --save_dir=./checkpoints \
    --save_every=10 \
    --validate_every=10
