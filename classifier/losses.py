import tensorflow as tf

def compute_hinge_loss(outputs, labels):
    total_loss = tf.losses.hinge_loss(tf.one_hot(labels,10),logits=outputs)
    mean_loss = tf.reduce_mean(total_loss)
    return mean_loss

def compute_softmax_loss(outputs, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=tf.one_hot(labels, 10)))

def compute_accuracy(outputs, labels):
    correct = tf.equal(tf.cast(tf.argmax(outputs, 1), tf.uint8 ), labels)
    return tf.reduce_mean(tf.cast(correct, tf.float32))