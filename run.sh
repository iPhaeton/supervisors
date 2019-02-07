python jobs.py \
    --job_name=siamese \
    --source_path=../input/mars/bbox_train/ \
    --model_path=../input/models/deep_sort_cnn/mars-small128.pb \
    --use_graph_creator=True \
    --batch_size=100 \
    --num_iter=100 \
    --num_per_class=4 \
    --margin=0.2 \
    --lr=1e-4