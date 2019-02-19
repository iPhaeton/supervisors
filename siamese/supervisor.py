import tensorflow as tf

import sys
sys.path.append("..")
from decorators import with_tensorboard, with_saver
from constants import ON_EPOCH_END, ON_LOG
from utils.metrics import l2_normalized
from siamese.losses.utils import mean_distances

def create_graph(session, base_model, optimizer, loss_fn, is_pretrained, normalized=True):
    """
    Creates graph for a siamese model
    
    Parameters:
    -----------
    - session: Tensorflow Session instance
    - base_model: tuple
        Sould contain two tensors (base_model input, base_model output).
    - optimizer: Tensorflow optimizer instance
    - loss_fn: Function
        Loss function
    - is_pretrained: bool
        If the model uses pretrained weights. If it does, only optimizer will be initialized.
    - normalized: bool
        Whether embeddings should be normalized before calculating loss.

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
        if normalized == True:
            outputs = l2_normalized(outputs)
        
        loss = loss_fn(labels=labels, embeddings=outputs)
        positive_mean_distance, negative_mean_distance = mean_distances(outputs, labels, metric=loss_fn.metric, normalized=normalized)
    
    with tf.name_scope('train_step'):
        train_step = optimizer.minimize(loss)
    
    if is_pretrained == True:
        session.run(tf.variables_initializer(optimizer.variables()))
    return inputs, outputs, labels, loss, positive_mean_distance, negative_mean_distance, train_step

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

@with_saver
@with_tensorboard
def train_siamese_model(
    session,
    model, 
    dirs,
    labels,
    batch_generator,
    batch_loader, 
    is_pretrained,
    epochs=100,
    observer=None,
    log_every=5,
    validate_every=5,
    batch_size=None,
    **kwargs,
):
    """
    Trains a siamese model
    
    Parameters:
    -----------
    - session: Tensorflow Session instance.
    - model: tuple
        Should contain tensors (inputs, outputs, labels, loss, train_step), described in create_graph function.
    - dirs: [[str]]
        List of training and validation directories [train_dirs, val_dirs]
    - labels: [[number]]
        List of training and class_labels [train_labels, val_labels]
    - batch_generator: iterator
        Sould yield [iteration, samples, batch_lables]. For the last batch in an epoch iteration == -1.
    - batch_loader: Function
        Should take dirs, labels, batch_size, random as parameters
        and return an iterator [samples, batch_lables]
    - is_pretrained: bool
        If the model is pretrained
    - epochs: int
        Number of epochs.
    - observer: EventAggregator
    - log_every: int
        Number of iterations between logs.
    - validate_every: int
        Number of iterations between validations.
    - batch_size: int
        Number of classes to use in a single batch.
    Returns:
    --------
    None
    """

    train_dirs, val_dirs = dirs
    train_labels, val_labels = labels
    
    inputs, outputs, labels, loss, positive_mean_distance, negative_mean_distance, train_step = model
    training_loss_summary = tf.summary.scalar("training_loss", loss)
    tarining_positive_mean_distance_summary = tf.summary.scalar('training_positive_mean_distance', positive_mean_distance)
    tarining_negative_mean_distance_summary = tf.summary.scalar('training_negative_mean_distance', negative_mean_distance)
    validation_loss_summary = tf.summary.scalar("validation_loss", loss)
    validation_positive_mean_distance_summary = tf.summary.scalar('validation_positive_mean_distance', positive_mean_distance)
    validation_negative_mean_distance_summary = tf.summary.scalar('validation_negative_mean_distance', negative_mean_distance)

    if is_pretrained == False:
        session.run(tf.global_variables_initializer())
    
    for i in range(epochs):
        for j, samples, batch_labels in batch_generator:
            feed_dict = {
                inputs: samples,
                labels: batch_labels,
            }

            batch_loss, _ = session.run([loss, train_step], feed_dict)

            print(f'Epoch {i}. Iteration {j}. Batch loss: {batch_loss}')
            
            if j == -1:
                break

        if (observer != None) & (i % log_every == 0):
            print('Calculating training loss...')
            log_samples, log_lables = batch_loader(dirs=train_dirs, labels=train_labels, random=True, batch_size=batch_size)
            observer.emit(ON_LOG, i, 
                {
                    inputs: log_samples,
                    labels: log_lables,
                }, 
                [
                    training_loss_summary, 
                    tarining_positive_mean_distance_summary, 
                    tarining_negative_mean_distance_summary
                ],
            )

        if (observer != None) & (i % validate_every == 0):
            print('Validating...')
            log_samples, log_lables = batch_loader(dirs=val_dirs, labels=val_labels, random=True, batch_size=None)
            observer.emit(ON_LOG, i, 
                {
                    inputs: log_samples,
                    labels: log_lables,
                }, 
                [
                    validation_loss_summary, 
                    validation_positive_mean_distance_summary, 
                    validation_negative_mean_distance_summary
                ],
            )

        if observer != None:
            observer.emit(ON_EPOCH_END, i, feed_dict)
        