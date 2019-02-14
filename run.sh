python jobs.py \
    --job_name=siamese \
    --source_path=../input/mars/bbox_train/ \
    --model_path=../input/models/deep_sort_cnn \
    --use_graph_creator=True \
    --batch_size=30 \
    --num_iter=100 \
    --num_per_class=4 \
    --margin=0.5 \
    --lr=1e-3

# python jobs.py \
#     --job_name=classifier \
#     --source_path=../input/cifar-10-batches-py \
#     --model_name=complex \
#     --loss=softmax \
#     --data=cifar10 \
#     --batch_size=64 \
#     --num_iter=1 \
#     --lr=1e-4
