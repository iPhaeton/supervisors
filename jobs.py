import tensorflow as tf
from pyramda import compose
from utils.curried_functions import tf_add, tf_cast, tf_multiply, filter_list
from utils.common import classes_to_labels
import os
from sklearn.model_selection import train_test_split
from loaders.data import load_CIFAR10_data
from loaders.batch import load_batch_of_images, cv2_loader, pil_loader, load_batch_of_data
from loaders.models import load_deep_sort_cnn, load_simple_model, load_simpler_model,load_complex_model
from utils.metrics import cosine_distance, eucledian_distance
from siamese.supervisor import train_siamese_model, create_graph as create_siamese_graph
from siamese.losses import triplet_semihard_loss
from classifier.supervisor import train_classifier, create_graph as create_classification_graph
from classifier.losses import compute_hinge_loss, compute_softmax_loss
import argparse
from constants import LOG_DIR_PATH
from auxillaries.events import EventAggregator
from functools import partial

def log_args(args):
    print('-----------------------------')
    print('Starting job with parameters:')
    for key, value in args.__dict__.items():
        print(f'--{key}={value}')
    print('-----------------------------')

def classification_job(source_path, **kwargs):

    model_loader = kwargs.pop('model_loader', None)
    graph_creator = kwargs.pop('graph_creator', None)
    loss_fn = kwargs.pop('loss_fn', None)
    data_loader = kwargs.pop('data_loader', None)
    batch_size = kwargs.pop('batch_size', None)
    num_iter = kwargs.pop('num_iter', 100)
    lr = kwargs.pop('lr', 1e-3)
    observer = kwargs.pop('observer', None)

    tf.reset_default_graph()

    inputs, outputs = model_loader()
    model = create_classification_graph(
        base_model=[inputs, outputs], 
        loss_fn=loss_fn,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
    )
    session = tf.Session()

    train_classifier(
        session=session,
        model=model,
        source_path=source_path,
        data_loader=data_loader,
        batch_loader=load_batch_of_data,
        num_iter=num_iter,
        batch_size=batch_size,
        observer=observer,
    )

def siamese_job(source_path, model_loader, **kwargs):
    loss_fn = kwargs.pop('loss_fn', None)
    batch_size = kwargs.pop('batch_size', None)
    num_iter = kwargs.pop('num_iter', 100)
    num_per_class = kwargs.pop('num_per_class', 5)
    margin = kwargs.pop('margin', 0.2)
    lr = kwargs.pop('lr', 1e-3)
    observer = kwargs.pop('observer', None)
    is_pretrained = kwargs.pop('is_pretrained', None)
    
    tf.reset_default_graph()

    dirs = compose(
        filter_list(['.DS_Store'], False),
        os.listdir,
    )(source_path)

    labels = classes_to_labels(dirs)
    #train_dirs, val_dirs, train_labels, val_labels = train_test_split(dirs, labels, test_size=0.1)

    session = tf.Session()
    inputs, outputs, is_pretrained = model_loader(session)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    model = create_siamese_graph(session=session, base_model=[inputs, outputs], optimizer=optimizer, loss_fn=loss_fn, is_pretrained=is_pretrained)

    train_siamese_model(
        session=session,
        model=model,
        batch_loader=load_batch_of_images(
            path=source_path, 
            path_dirs=dirs[0:30], 
            labels=labels[0:30], 
            num_per_class=num_per_class, 
            batch_size=None,
            image_shape=(128, 64, 3), 
            loader=cv2_loader,
        ),
        num_iter=num_iter,
        observer=observer,
        log_dir=LOG_DIR_PATH,
        log_every=5,
        is_pretrained=is_pretrained,
    )

def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Freeze old model")
    parser.add_argument(
        "--job_name",
        required=True,
        choices=['siamese', 'classifier'],
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
        default=None,
        help="Path to the model",
        type=str,
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Path to the checkpoint",
        type=str,
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
        "--metric",
        default='eucledian',
        help="Metric for simese model loss",
        type=str,
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
    parser.add_argument(
        "--model_name",
        default=None,
        help="Model name",
    )
    parser.add_argument(
        "--loss",
        default=None,
        help="Name of loass function",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Data name",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    log_args(args)

    observer = EventAggregator()

    #get model loader
    if args.model_name == 'simple':
        model_loader = load_simple_model
    elif args.model_name == 'simpler':
        model_loader = load_simpler_model
    elif args.model_name == 'complex':
        model_loader = load_complex_model
    elif args.model_name == 'deep_sort_cnn':
        model_loader = load_deep_sort_cnn(
            model_path=args.model_path, 
            checkpoint_path=args.checkpoint_path,
        )

    #get metric
    if args.metric == 'eucledian':
        metric = partial(eucledian_distance, squared=True)
    elif args.metric == 'cosine':
        metric = cosine_distance

    #get loss function
    if args.loss == 'hinge':
        loss_fn = compute_hinge_loss
    elif args.loss == 'softmax':
        loss_fn = compute_softmax_loss
    elif args.loss == 'triplet_semihard':
        loss_fn = partial(triplet_semihard_loss, metric=metric, margin=args.margin)

    #get data loader
    if args.data == 'cifar10':
        data_loader = load_CIFAR10_data

    #run job
    if args.job_name == 'classifier':
        classification_job(
            args.source_path,
            model_loader=model_loader,
            graph_creator=create_classification_graph,
            loss_fn=loss_fn,
            data_loader=data_loader,
            batch_size=args.batch_size,
            num_iter=args.num_iter,
            lr=args.lr,
            observer=observer,
        )
    elif args.job_name == 'siamese':
        siamese_job(
            args.source_path, 
            model_loader,
            loss_fn=loss_fn,
            batch_size=args.batch_size, 
            num_iter=args.num_iter,
            num_per_class=args.num_per_class,
            margin=args.margin,
            lr=args.lr,
            observer=observer,
        )

if __name__ == '__main__':
    main()