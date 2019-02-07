#### Original implementation: https://omoindrot.github.io/triplet-loss

import tensorflow as tf
from pyramda import compose
from utils.curried_functions import tf_cast, tf_add, tf_multiply
import numpy as np

def get_positive_mask(labels):
    """
    Parameters:
    -----------
    - labels: [int]
        List of labels of size N (number of samples).
    Returns:
    ----------
    - positive_mask: Tensor (N, N)
        A square martix with True for all positive samples and False for all negative samples.
    """
    return tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

get_not_anchor_mask = compose(
    tf.logical_not,
    tf_cast(dtype = tf.bool),
    tf.eye,
    lambda shape: shape[0],
    tf.shape,
)

get_not_anchor_mask.__doc__ = """
    Parameters:
    -----------
    - labels: [int]
        List of labels of size N (number of samples).
    Returns:
    ----------
    - not_anchor_mask: Tensor (N, N)
        A square martix with False for all anchors and True for other samples, like
        [[0 1 1]
         [1 0 1]
         [1 1 0]]
"""

def get_anchor_positive_mask(labels):
    """
    Parameters:
    -----------
    - labels: [int]
        List of labels of size N (number of samples).
    Returns:
    ----------
    - anchor_positive_mask: Tensor (N, N)
        A square martix with ones for all positive samples, except anchors on main diagonal, 
        and zeros for all other samples.
    """
    return tf.to_float(
        tf.logical_and(
            get_not_anchor_mask(labels),
            get_positive_mask(labels),
        )
    )

get_negative_mask = compose(
    tf_add(1.),
    tf_multiply(np.finfo(np.float32).max),
    tf.to_float,
    get_positive_mask,
)

get_negative_mask.__doc__ = """
    Parameters:
    -----------
    - labels: [int]
        List of labels of size N (number of samples).
    Returns:
    ----------
    - positive_mask: Tensor (N, N)
        A square martix with ones for all negative samples and infinity for all positive samples.
"""

def compute_loss(model, metric, masks, margin):
    """
    Compute triplet loss
    
    Parameters:
    -----------
    - model: tuple
        Model input tensor and model output tensor.
    - metric: Function
        Should take output tensor as a parameter and compute distance matrix between outputs.
    - masks: tuple
        Contains two matrices: 
            a square martix with ones for all positive samples, except anchors on main diagonal, 
            and zeros for all other samples;
            a square martix with ones for all negative samples and infinity 
            for all positive samples.
    - margin: float
        Minimum margin between positive and negative distance.
    """
    inputs, outputs = model
    anchor_positive_mask, negative_mask = masks

    distances = metric(outputs)
    positive_distances = tf.multiply(anchor_positive_mask, distances)
    negative_distances = tf.multiply(negative_mask, distances)

    loss = tf.expand_dims(positive_distances, 2) - tf.expand_dims(negative_distances, 1) + margin
    loss = tf.maximum(loss, 0.)
    
    num_triplets = compose(
        tf.reduce_sum,
        tf.to_float,
    )(tf.greater(loss, 0.))
    
    loss = tf.reduce_sum(loss) / (num_triplets + 1e-16)

    return loss

def validate_siamese_model(
    session,
    model, 
    source_path, 
    val_dirs, 
    val_labels,
    metric,
    margin,
    batch_loader,
    num_per_class,
    batch_size,
):
    """
    Validates a siamese model
    
    Parameters:
    -----------
    - session: Tensorflow Session instance.
    - model: tuple
        Sould contain three tensors (inputs, labels, loss).
    - source_path: string
        Path to model data.
    - val_dirs: [string]
        List of validation directories.
    - val_labels: [int]
        List of validation class labels.
    - metric: Function
        Should take output tensor as a parameter and compute distance matrix between outputs.
    - batch_loader: Function
        Should take source_path, train_dirs, train_labels, num_per_class as parameters
        and return an iterator [samples, batch_lables]
    - num_per_class: int
        Number of samples randomly chosen from each class
    - batch_size: int
        Number of classes to use in a single batch. Total number of samples will be batch_size * num_per_class
    
    Returns:
    --------
    - batch_loss: float
        Value of the loss function on the validation batch.
    """


    samples, batch_lables = batch_loader(source_path, val_dirs, val_labels, num_per_class, batch_size if (batch_size != None) and (batch_size < len(val_dirs)) else None)
    inputs, labels, loss = model
    
    batch_loss= session.run(loss, {
        inputs: samples,
        labels: batch_lables,
    })

    return batch_loss

def train_siamese_model(
    session,
    model, 
    source_path, 
    dirs, 
    class_labels, 
    metric, 
    optimizer,
    batch_loader,
    margin=0.2, 
    num_per_class=5, 
    num_iter=1000,
    batch_size=None,
):
    """
    Trains a siamese model
    
    Parameters:
    -----------
    - session: Tensorflow Session instance.
    - model: tuple
        Sould contain two tensors (model input, model output).
    - source_path: string
        Path to model data.
    - dirs: [[string], [string]]
        Lists of training and validation directories.
    - class_labels: [[int], [int]]
        Lists of training and validation class labels.
    - metric: Function
        Should take output tensor as a parameter and compute distance matrix between outputs.
    - optimizer: Tensorflow optimizer instance
    - batch_loader: Function
        Should take source_path, train_dirs, train_labels, num_per_class as parameters
        and return an iterator [samples, batch_lables]
    - margin: float
        Desired margin between negative and positive samples.
    - num_per_class: int
        Number of samples randomly chosen from each class
    - num_iter: int
        Number of iterations
    - batch_size: int
        Number of classes to use in a single batch. Total number of samples will be batch_size * num_per_class

    Returns:
    --------
    None
    """
    
    train_dirs, val_dirs = dirs
    train_labels, val_labels = class_labels
    
    inputs, outputs = model
    labels = tf.placeholder(name='labels', dtype=tf.int8)
    anchor_positive_mask = get_anchor_positive_mask(labels)
    negetive_mask = get_negative_mask(labels)
    
    loss = compute_loss(
        model=(inputs, outputs), 
        metric=metric, 
        masks=(anchor_positive_mask, negetive_mask), 
        margin=margin,
    )
    
    train_step = optimizer.minimize(loss)
    session.run(tf.global_variables_initializer())
    
    for i in range(num_iter):
        samples, batch_lables = batch_loader(source_path, train_dirs, train_labels, num_per_class, batch_size)
        batch_loss, _ = session.run([loss, train_step], {
            inputs: samples,
            labels: batch_lables,
        })

        val_loss = validate_siamese_model(
            session=session, 
            model=[inputs, labels, loss], 
            source_path=source_path, 
            val_dirs=val_dirs,
            val_labels=val_labels,
            metric=metric,
            margin=margin,
            batch_loader=batch_loader,
            num_per_class=num_per_class,
            batch_size=batch_size,
        )

        print(f'{{"metric": "Train loss", "value"{batch_loss}}}')
        print(f'{{"metric": "Val loss", "value"{val_loss}}}')