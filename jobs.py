import tensorflow as tf
from pyramda import compose
from utils.curried_functions import tf_add, tf_cast, tf_multiply, filter_list
from utils.common import classes_to_labels
import os
from sklearn.model_selection import train_test_split
from loaders import load_batch_of_images, load_model_pb, cv2_loader
from utils.metrics import cosine_distance
from siamese import train_siamese_model, create_graph as create_siamese_graph
import argparse
from constants import LOG_DIR_PATH

import sys
sys.path.append("..")
from input.models.deep_sort_cnn.freeze_model import create_graph

def log_args(args):
    print('-----------------------------')
    print('Starting job with parameters:')
    for key, value in args.__dict__.items():
        print(f'--{key}={value}')
    print('-----------------------------')

def siamese_job(source_path, model_path, **kwargs):

    graph_creator = kwargs.pop('graph_creator', None)
    batch_size = kwargs.pop('batch_size', None)
    num_iter = kwargs.pop('num_iter', 100)
    num_per_class = kwargs.pop('num_per_class', 5)
    margin = kwargs.pop('margin', 0.2)
    lr = kwargs.pop('lr', 1e-3)
    
    tf.reset_default_graph()

    dirs = compose(
        filter_list(['.DS_Store'], False),
        os.listdir,
    )(source_path)

    labels = classes_to_labels(dirs)
    train_dirs, val_dirs, train_labels, val_labels = train_test_split(dirs, labels, test_size=0.1)

    inputs, outputs, _ = load_model_pb(
        model_path, 
        input_name="images", 
        output_name="features", 
        graph_creator=graph_creator,
    )

    model = create_siamese_graph(base_model=[inputs, outputs], metric=cosine_distance, margin=margin, optimizer=tf.train.AdamOptimizer(learning_rate=lr),)
    session = tf.Session()

    train_siamese_model(
        session=session,
        model=model,
        source_path=source_path,
        dirs=(train_dirs, val_dirs),
        class_labels=(train_labels, val_labels),
        metric=cosine_distance,
        batch_loader=load_batch_of_images(image_shape=(128, 64, 3), loader=cv2_loader),
        margin=margin,
        num_iter=num_iter,
        num_per_class=num_per_class,
        batch_size=batch_size,
        log_dir=LOG_DIR_PATH,
    )

def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Freeze old model")
    parser.add_argument(
        "--job_name",
        required=True,
        choices=['siamese'],
        type=str,
    )
    parser.add_argument(
        "--source_path",
        required=True,
        help="Path to the data",
        type=str,
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the model",
        type=str,
    )
    parser.add_argument(
        "--use_graph_creator",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--batch_size",
        default=None,
        type=int
    )
    parser.add_argument(
        "--num_iter",
        default=100,
        help='Number of iterations',
        type=int
    )
    parser.add_argument(
        "--num_per_class",
        default=5,
        help="Number of samples per class in each batch",
        type=int
    )
    parser.add_argument(
        "--margin",
        default=0.2,
        help="Desired margin between positive and negarive distances",
        type=float,
    )
    parser.add_argument(
        "--lr",
        default=1e-3,
        help="Learning rate",
        type=float,
    )
    return parser.parse_args()

def main():
    args = parse_args()
    log_args(args)

    if args.job_name == 'siamese':
        siamese_job(
            args.source_path, 
            args.model_path, 
            graph_creator=create_graph if args.use_graph_creator else None, 
            batch_size=args.batch_size, 
            num_iter=args.num_iter,
            num_per_class=args.num_per_class,
            margin=args.margin,
            lr=args.lr,
        )

if __name__ == '__main__':
    main()