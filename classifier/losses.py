import tensorflow as tf

def compute_hinge_loss(outputs, labels):
    total_loss = tf.losses.hinge_loss(tf.one_hot(y,10),logits=outputs)
    mean_loss = tf.reduce_mean(total_loss)
    return mean_loss