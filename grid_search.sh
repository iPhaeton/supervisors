for loss in triplet_semihard triplet_hard triplet_all;
do
    for metric in cosine eucledian euclefdian_squared;
    do
        for normalized in 0 1;
            do
            for lr in 1e-2 1e-3 1e-4;
            do
                #floyd
                python jobs.py \
                --job_name=siamese \
                --model_name=deep_sort_cnn \
                --source_path=../input/mars/bbox_train/ \
                --model_path=../input/models/deep_sort_cnn/freeze_model.py \
                --checkpoint_path=../input/models/deep_sort_cnn/mars-small128.ckpt-68577 \
                --batch_size=32 \
                --epochs=15 \
                --num_per_class=12 \
                --loss=$loss \
                --metric=$metric \
                --margin=0.5 \
                --lr=$lr \
                --log_every=1 \
                --save_dir=./checkpoints \
                --save_every=1000 \
                --validate_every=1 \
                --normalized=$normalized \
                --log_dir="./logs/log-${loss}-${metric}-${normalized}-${lr}"

                #local
                # python jobs.py \
                #     --job_name=siamese \
                #     --model_name=deep_sort_cnn \
                #     --source_path=../input/mars/bbox_train/ \
                #     --model_path=../input/models/deep_sort_cnn/freeze_model.py \
                #     --checkpoint_path=../input/models/deep_sort_cnn/mars-small128.ckpt-68577 \
                #     --batch_size=10 \
                #     --epochs=11 \
                #     --num_per_class=4 \
                #     --loss=$loss \
                #     --metric=$metric \
                #     --margin=0.5 \
                #     --lr=$lr \
                #     --log_every=2 \
                #     --save_dir=./checkpoints \
                #     --save_every=5 \
                #     --validate_every=3 \
                #     --normalized=$normalized \
                #     --log_dir="./logs/log-${loss}-${metric}-${normalized}-${lr}"
            done
        done
    done
done