import tensorflow as tf
import numpy as np
import math

import sys
sys.path.append("..")
from classifier.losses import compute_hinge_loss, compute_accuracy
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
    source_path,
    data_loader,
    num_iter=100,
    batch_size=None,
    observer=None,
):
    inputs, outputs, labels, loss, train_step = model

    accuracy = compute_accuracy(outputs, labels)

    session.run(tf.global_variables_initializer())
    samples, data_labels, samples_val, labels_val, samples_test, labels_test = data_loader(source_path)

    iter_count = 0
    for i in range(num_iter):
        indices = np.random.permutation(range(samples.shape[0]))
        for j in range(int(math.ceil(samples.shape[0]/batch_size))):
            start_idx = j * batch_size
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx : end_idx]

            batch = samples[batch_indices,:,:,:]
            batch_labels = data_labels[batch_indices]

            feed_dict = {
                inputs: batch,
                labels: batch_labels,
            }

            if observer != None:
                observer.emit(ON_ITER_START, i, feed_dict)
            
            batch_loss, batch_accuracy, _ = session.run([loss, accuracy, train_step], feed_dict)
            
            if observer != None:
                observer.emit(ON_ITER_END, i, feed_dict)

            if iter_count % 100 == 0:
                print(f'{iter_count}: {batch_loss}; {batch_accuracy}')

            iter_count +=1