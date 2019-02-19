python jobs.py \
    --job_name=siamese \
    --model_name=deep_sort_cnn \
    --source_path=../input/mars/bbox_train/ \
    --model_path=../input/models/deep_sort_cnn/freeze_model.py \
    --checkpoint_path=../input/models/deep_sort_cnn/mars-small128.ckpt-68577 \
    --batch_size=10 \
    --epochs=11 \
    --num_per_class=4 \
    --loss=triplet_hard \
    --metric=eucledian \
    --margin=0.5 \
    --lr=1e-3 \
    --log_every=2 \
    --save_dir=./checkpoints \
    --save_every=5 \
    --validate_every=3 \
    --normalized=1 \

# python jobs.py \
#     --job_name=classifier \
#     --source_path=../input/cifar-10-batches-py \
#     --model_name=complex \
#     --loss=softmax \
#     --data=cifar10 \
#     --batch_size=64 \
#     --num_iter=1 \
#     --lr=1e-4
