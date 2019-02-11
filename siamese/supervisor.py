import tensorflow as tf

import sys
sys.path.append("..")
from decorators import with_tensorboard
from constants import ON_ITER_START, ON_ITER_END
from siamese.triplet_loss import get_anchor_positive_mask, get_negative_mask, compute_loss

def create_graph(base_model, metric, margin, optimizer):
    """
    Creates graph for a siamese model
    
    Parameters:
    -----------
    - base_model: tuple
        Sould contain two tensors (base_model input, base_model output).
    - metric: Function
        Should take output tensor as a parameter and compute distance matrix between outputs.
    - margin: float
        Desired margin between negative and positive samples.
    - optimizer: Tensorflow optimizer instance

    Returns:
    --------
    - inputs: Tensor
        Model inputs
    - outputs: Tensor
        Model outputs
    - labels: Tensor
        Placeholder for labels.
    - loss: Tensor
        Computed loss function.
    - train_step
        Loss minimizer.
    """

    with tf.name_scope('base_model'):
        inputs, outputs = base_model
    
    with tf.name_scope('loss'):
        labels = tf.placeholder(name='labels', dtype=tf.int32, shape=(None,))
        anchor_positive_mask = get_anchor_positive_mask(labels)
        negetive_mask = get_negative_mask(labels)

        # loss = compute_loss(
        #     model=(inputs, outputs), 
        #     metric=metric, 
        #     masks=(anchor_positive_mask, negetive_mask), 
        #     margin=margin,
        # )
        
        loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(
            labels=labels,
            embeddings=outputs,
            margin=margin,
        )
    
    with tf.name_scope('train_step'):
        train_step = optimizer.minimize(loss)

    return inputs, outputs, labels, loss, train_step

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

@with_tensorboard
def train_siamese_model(
    session,
    model, 
    source_path, 
    dirs, 
    class_labels, 
    metric, 
    batch_loader,
    margin=0.2, 
    num_per_class=5, 
    num_iter=100,
    batch_size=None,
    observer=None,
):
    """
    Trains a siamese model
    
    Parameters:
    -----------
    - session: Tensorflow Session instance.
    - model: tuple
        Sould contain tensors (inputs, outputs, labels, loss, train_step), described in create_graph function.
    - source_path: string
        Path to model data.
    - dirs: [[string], [string]]
        Lists of training and validation directories.
    - class_labels: [[int], [int]]
        Lists of training and validation class labels.
    - metric: Function
        Should take output tensor as a parameter and compute distance matrix between outputs.
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
    - observer: EventAggregator
    Returns:
    --------
    None
    """
    
    train_dirs, val_dirs = dirs
    train_labels, val_labels = class_labels
    inputs, outputs, labels, loss, train_step = model
   
    session.run(tf.global_variables_initializer())
    
    for i in range(num_iter):
        samples, batch_labels = batch_loader(source_path, train_dirs, train_labels, num_per_class, batch_size)
        feed_dict = {
            inputs: samples,
            labels: batch_labels,
        }

        if observer != None:
            observer.emit(ON_ITER_START, i, feed_dict)

        batch_loss, _ = session.run([loss, train_step], feed_dict)

        if observer != None:
            observer.emit(ON_ITER_END, i, feed_dict)

        # val_loss = validate_siamese_model(
        #     session=session, 
        #     model=[inputs, labels, loss], 
        #     source_path=source_path, 
        #     val_dirs=val_dirs,
        #     val_labels=val_labels,
        #     metric=metric,
        #     margin=margin,
        #     batch_loader=batch_loader,
        #     num_per_class=num_per_class,
        #     batch_size=batch_size,
        # )

        print(f'{{"metric": "Train loss", "value"{batch_loss}}}')
        # print(f'{{"metric": "Val loss", "value"{val_loss}}}')