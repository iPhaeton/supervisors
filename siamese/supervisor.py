import tensorflow as tf

import sys
sys.path.append("..")
from decorators import with_tensorboard
from constants import ON_ITER_START, ON_ITER_END

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
        tf.summary.scalar('loss', loss)
    
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

@with_tensorboard
def train_siamese_model(
    session,
    model, 
    batch_loader, 
    is_pretrained,
    num_iter=100,
    observer=None,
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
    - num_iter: int
        Number of iterations
    - observer: EventAggregator
    Returns:
    --------
    None
    """
    
    inputs, outputs, labels, loss, train_step = model
    if is_pretrained == False:
        session.run(tf.global_variables_initializer())
    
    for i in range(num_iter):
        samples, batch_labels = batch_loader()
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