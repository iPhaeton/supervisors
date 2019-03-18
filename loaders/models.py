import tensorflow as tf
import tensorflow.contrib.slim as slim
import importlib.util
from decorators import partially_applied

@partially_applied
def load_deep_sort_cnn(session, model_path, checkpoint_path):
    """
    Load a model from saved .pb file and checkpoint
    
    Parameters:
    -----------
    - session: Tensorflow Session instance
    - model_path: string
        Path to the model definition .py file.
    - checkpoint_path: string
        Path to the model checkpoint file
        
    Returns:
    --------
    - inputs: Tensor (N, H, W, C)
        Images
        N - number of samples (None)
        H - image height
        W - image width
        C - number of channels
    - outputs: Tensor (N, E)
        Image embeddings
        N - number of samples (None)
        E - embedding size
    """
    spec = importlib.util.spec_from_file_location("module.model", model_path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    create_graph = model.create_graph
    inputs, outputs = create_graph(session)
    
    if checkpoint_path != None:
        saver = tf.train.Saver(slim.get_variables_to_restore())
        saver.restore(session, checkpoint_path)

    return inputs, outputs, checkpoint_path != None

def load_simplest_model(_):
    X = tf.placeholder(tf.float32, [None, 64, 64, 3])
    X_flat = tf.reshape(X,[-1,12288])

    W0 = tf.get_variable("W0", shape=[12288, 128])
    tf.summary.histogram('W0', W0)
    a_out = tf.matmul(X_flat,W0)
    y_out = tf.nn.sigmoid(a_out)
    return X, y_out, False


def load_simple_model(_):
    X = tf.placeholder(tf.float32, [None, 64, 64, 3])
    # setup variables
    Wconv0 = tf.get_variable("Wconv0", shape=[2, 2, 3, 16])
    bconv0 = tf.get_variable("bconv0", shape=[16])
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 16, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    W1 = tf.get_variable("W1", shape=[5408, 128])
    b1 = tf.get_variable("b1", shape=[128])

    tf.summary.histogram('Wconv0', Wconv0)
    tf.summary.histogram('bconv0', bconv0)
    tf.summary.histogram('Wconv1', Wconv1)
    tf.summary.histogram('bconv1', bconv1)
    tf.summary.histogram('W1', W1)
    tf.summary.histogram('b1', b1)

    # define our graph (e.g. two_layer_convnet)
    a0 = tf.nn.conv2d(X, Wconv0, strides=[1,2,2,1], padding='VALID') + bconv0
    h0 = tf.nn.relu(a0)
    a1 = tf.nn.conv2d(h0, Wconv1, strides=[1,2,2,1], padding='VALID') + bconv1
    h1 = tf.nn.relu(a1)
    h1_flat = tf.reshape(h1,[-1,5408])
    a_out = tf.matmul(h1_flat,W1) + b1
    y_out = tf.nn.relu(a_out)
    return X, y_out, False

def load_simpler_model():
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    # setup variables
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    W1 = tf.get_variable("W1", shape=[5408, 10])
    b1 = tf.get_variable("b1", shape=[10])

    # define our graph (e.g. two_layer_convnet)
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1,2,2,1], padding='VALID') + bconv1
    h1 = tf.nn.relu(a1)
    h1_flat = tf.reshape(h1,[-1,5408])
    y_out = tf.matmul(h1_flat,W1) + b1
    return X, y_out

def load_complex_model():
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])

    W1 = tf.get_variable('W1', shape=[7,7,3,32])
    b1 = tf.get_variable('b1', shape=[32])
    beta1 = tf.get_variable('beta1', shape=[32])
    gamma1 = tf.get_variable('gamma1', shape=[32])
    W2 = tf.get_variable('W2', shape=[5408,1024])
    b2 = tf.get_variable('b2', shape=[1024])
    W3 = tf.get_variable('W3', shape=[1024,10])
    b3 = tf.get_variable('b3', shape=[10])
    
    a1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='VALID') + b1
    h1 = tf.nn.relu(a1)
    h1_mean, h1_var = tf.nn.moments(h1, axes=[0,1,2])
    h1_norm = tf.nn.batch_normalization(h1, mean=h1_mean, variance=h1_var, offset=beta1, scale=gamma1, variance_epsilon=1e-5)
    h1_pool = tf.nn.max_pool(h1_norm, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    h1_flat = tf.reshape(h1_pool, [-1,5408])
    a2= tf.matmul(h1_flat, W2) + b2
    h2 = tf.nn.relu(a2)
    y_out = tf.matmul(h2, W3) + b3
    
    return X, y_out