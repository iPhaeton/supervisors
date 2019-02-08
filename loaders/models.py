import tensorflow as tf

def load_model_pb(checkpoint_filename, input_name, output_name, **kwargs):
    """
    Load a model from saved .pb file and checkpoint
    
    Parameters:
    -----------
    - checkpoint_filename: string
        Path to the checkpoint on disk.
    - input_name: string
        Name of the input variable in the graph.
    - output_name: string
        Name of the output variable in the graph.

    Keyword arguments:
    ------------------
    - graph_creator: Function
        Should create the model graph
        
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
    graph_creator = kwargs.pop('graph_creator', None)
    
    inputs, outputs = None, None
    if graph_creator != None:
        inputs, outputs = graph_creator()
    
    with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(file_handle.read())
    
    tf.import_graph_def(graph_def, name="net")
    
    if graph_creator == None:
        inputs = tf.get_default_graph().get_tensor_by_name("net/%s:0" % input_name)
        outputs = tf.get_default_graph().get_tensor_by_name("net/%s:0" % output_name)
    
    return inputs, outputs, graph_def

def load_simple_model():
    X = tf.placeholder(tf.float32, [None, 64, 64, 3])
    # setup variables
    Wconv0 = tf.get_variable("Wconv0", shape=[2, 2, 3, 16])
    bconv0 = tf.get_variable("bconv0", shape=[16])
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 16, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    W1 = tf.get_variable("W1", shape=[5408, 10])
    b1 = tf.get_variable("b1", shape=[10])

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
    y_out = tf.matmul(h1_flat,W1) + b1
    return X, y_out

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