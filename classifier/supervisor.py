import tensorflow as tf
import numpy as np

import sys
sys.path.append("..")
from classifier.losses import compute_hinge_loss
from constants import ON_ITER_START, ON_ITER_END

def create_graph(base_model, loss_fn, optimizer):
    with tf.name_scope('base_model'):
        inputs, outputs = base_model

    with tf.name_scope('loss'):
        labels = tf.placeholder(name='labels', dtype=tf.uint8)
        loss = loss_fn(outputs, labels)

    with tf.name_scope('train_step'):
        train_step = optimizer.minimize(loss)

    return inputs, outputs, labels, loss, train_step

def train_classifier(
    session,
    model, 
    data_loader,
    num_iter=100,
    batch_size=None,
    observer=None,
):
    inputs, outputs, labels, loss, train_step = model

    session.run(tf.global_variables_initializer())
    samples, data_labels = data_loader()

    for i in range(num_iter):
        batch = np.random.choice(samples, size=batch_size)
        batch_labels = np.random.choice(data_labels, size=batch_size)

        feed_dict = {
            inputs: samples,
            labels: batch_labels,
        }

        if observer != None:
            observer.emit(ON_ITER_START, i, feed_dict)

        batch_loss, _ = session.run([loss, train_step], feed_dict)

        if observer != None:
            observer.emit(ON_ITER_END, i, feed_dict)

        print(f'{{"metric": "Train loss", "value"{batch_loss}}}')