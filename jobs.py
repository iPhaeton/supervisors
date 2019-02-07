import tensorflow as tf
from pyramda import compose
from utils.curried_functions import tf_add, tf_cast, tf_multiply, filter_list
from utils.common import classes_to_labels
import os
from sklearn.model_selection import train_test_split
from loaders import load_batch_of_images, load_model_pb
from utils.metrics import cosine_distance
from siamese import train_siamese_model

def siamese_job(source_path, model_path, **kwargs):

    graph_creator = kwargs.pop('graph_creator', None)

    tf.reset_default_graph()

    dirs = compose(
        filter_list(['.DS_Store'], False),
        os.listdir,
    )(source_path)

    labels = classes_to_labels(dirs)
    train_dirs, val_dirs, train_labels, val_labels = train_test_split(dirs[0:20], labels[0:20], test_size=0.1)

    inputs, outputs, _ = load_model_pb(
        model_path, 
        input_name="images", 
        output_name="features", 
        graph_creator=graph_creator,
    )

    session = tf.Session()

    train_siamese_model(
        session=session,
        model=[inputs, outputs],
        source_path=source_path,
        dirs=(train_dirs, val_dirs),
        class_labels=(train_labels, val_labels),
        metric=cosine_distance,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.00001),
        batch_loader=load_batch_of_images(image_shape=(128, 64, 3)),
        num_iter=2,
    )

    