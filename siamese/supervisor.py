import tensorflow as tf

import sys
sys.path.append("..")
from decorators import with_tensorboard, with_saver, with_validator
from constants import ON_ITER_START, ON_ITER_END, ON_LOG, ON_VALIDATION

def create_graph(session, base_model, optimizer, loss_fn, is_pretrained):
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
        loss = loss_fn(labels=labels, embeddings=outputs)
        #tf.summary.scalar('loss', loss)
    
    with tf.name_scope('train_step'):
        train_step = optimizer.minimize(loss)
    
    if is_pretrained == True:
        session.run(tf.variables_initializer(optimizer.variables()))
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

@with_saver
@with_tensorboard
#@with_validator
def train_siamese_model(
    session,
    model, 
    batch_loader, 
    is_pretrained,
    epochs=100,
    observer=None,
    **kwargs,
):
    """
    Trains a siamese model
    
    Parameters:
    -----------
    - session: Tensorflow Session instance.
    - model: tuple
        Sould contain tensors (inputs, outputs, labels, loss, train_step), described in create_graph function.
    - batch_loader: Function
        Should take source_path, train_dirs, train_labels, num_per_class as parameters
        and return an iterator [samples, batch_lables]
    - is_pretrained: bool
        If the model is pretrained
    - epochs: int
        Number of epochs.
    - observer: EventAggregator
    Returns:
    --------
    None
    """
    
    inputs, outputs, labels, loss, train_step = model
    training_summary = tf.summary.scalar("training_loss", loss)
    validation_summary = tf.summary.scalar("validation_loss", loss)
    if is_pretrained == False:
        session.run(tf.global_variables_initializer())
    
    for i in range(epochs):
        for j, samples, batch_labels in batch_loader:
            feed_dict = {
                inputs: samples,
                labels: batch_labels,
            }

            if observer != None:
                observer.emit(ON_LOG, i, feed_dict, [training_summary])
                observer.emit(ON_VALIDATION, i, [inputs, labels], validation_summary)

            batch_loss, _ = session.run([loss, train_step], feed_dict)

            if observer != None:
                observer.emit(ON_ITER_END, i, feed_dict)

            print(j, f'{{"metric": "Train loss", "value": "{batch_loss}"}}')
            
            if j == -1:
                break
        