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
        inputs, outputs, labels = base_model
    
    with tf.name_scope('loss'):
        if normalized == True:
            outputs = l2_normalized(outputs)
        
        loss, num_positive_triplets = loss_fn(labels=labels, embeddings=outputs)
        distance_metrics = mean_distances(outputs, labels, metric=loss_fn.metric, normalized=normalized)
    
    with tf.name_scope('train_step'):
        train_step = optimizer.minimize(loss)
    
    if is_pretrained == True:
        session.run(tf.variables_initializer(optimizer.variables()))
    
    return inputs, outputs, labels, loss, train_step, distance_metrics, num_positive_triplets

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
    
    inputs, outputs, labels, loss, train_step, distance_metrics, num_positive_triplets = model
    positive_mean_distance, negative_mean_distance, hardest_mean_positive_distance, hardest_mean_negative_distance = distance_metrics
    
    with tf.name_scope('training'):
        training_loss_summary = tf.summary.scalar("loss", loss)
        training_positive_mean_distance_summary = tf.summary.scalar("positive_mean_distance", positive_mean_distance)
        training_negative_mean_distance_summary = tf.summary.scalar("negative_mean_distance", negative_mean_distance)
        training_hardest_mean_positive_distance_summary = tf.summary.scalar("hardest_mean_positive_distance", hardest_mean_positive_distance)
        training_hardest_mean_negative_distance_summary = tf.summary.scalar("hardest_mean_negative_distance", hardest_mean_negative_distance)
        num_positive_triplets_summary = tf.summary.scalar("num_positive_triplets", num_positive_triplets)

    with tf.name_scope('validation'):
        validation_loss_summary = tf.summary.scalar("loss", loss)
        validation_positive_mean_distance_summary = tf.summary.scalar("positive_mean_distance", positive_mean_distance)
        validation_negative_mean_distance_summary = tf.summary.scalar("negative_mean_distance", negative_mean_distance)
        validation_hardest_mean_positive_distance_summary = tf.summary.scalar("hardest_mean_positive_distance", hardest_mean_positive_distance)
        validation_hardest_mean_negative_distance_summary = tf.summary.scalar("hardest_mean_negative_distance", hardest_mean_negative_distance)

    if is_pretrained == False:
        session.run(tf.global_variables_initializer())

    if observer != None:
            print('Calculating training loss...')
            log_samples, log_lables = batch_loader(dirs=train_dirs, labels=train_labels, random=True, batch_size=batch_size)
            observer.emit(ON_LOG, -1, 
                {
                    inputs: log_samples,
                    labels: log_lables,
                }, 
                [
                    training_loss_summary,
                    training_positive_mean_distance_summary,
                    training_negative_mean_distance_summary,
                    training_hardest_mean_positive_distance_summary,
                    training_hardest_mean_negative_distance_summary,
                    num_positive_triplets_summary,
                ],
            )

            print('Validating...')
            log_samples, log_lables = batch_loader(dirs=val_dirs, labels=val_labels, random=True, batch_size=None)
            observer.emit(ON_LOG, -1, 
                {
                    inputs: log_samples,
                    labels: log_lables,
                }, 
                [
                    validation_loss_summary,
                    validation_positive_mean_distance_summary,
                    validation_negative_mean_distance_summary,
                    validation_hardest_mean_positive_distance_summary,
                    validation_hardest_mean_negative_distance_summary,
                ],
            )
    
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
                    training_positive_mean_distance_summary,
                    training_negative_mean_distance_summary,
                    training_hardest_mean_positive_distance_summary,
                    training_hardest_mean_negative_distance_summary,
                    num_positive_triplets_summary,
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
                    validation_negative_mean_distance_summary,
                    validation_hardest_mean_positive_distance_summary,
                    validation_hardest_mean_negative_distance_summary,
                ],
            )

        if observer != None:
            observer.emit(ON_EPOCH_END, i, feed_dict)
        