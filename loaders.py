import tensorflow as tf
from PIL import Image
import numpy as np
import os
from pyramda import curry
from decorators import partially_applied
import cv2

#loads images in RGB format
def pil_loader(path, image_shape):
    img = Image.open(path)
    img = img.resize((image_shape[1], image_shape[0]))
    return np.array(img)

#loads images in BGR format
def cv2_loader(path, image_shape):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (image_shape[1], image_shape[0]))
    return np.array(img)

@partially_applied
def load_batch_of_images(path, path_dirs, labels, num_per_class, batch_size=None, **kwargs):
    """
    Loads a random batch of images
    
    Parameters:
    -----------
    - path: string
        Path to the image source directory on disk. 
        Source directory should be divided into directories, one directory per class.
    - path_dirs: [string]
        List of directories contained in the path (classes).
    - labels: [int]
        Class labels. Should correspond to classes.
    - num_per_class: int
        Number of images randomly chosen from each class
    - batch_size: int
        Number of classes to use in a single batch. Total number of samples will be batch_size * num_per_class

    Keyword arguments:
    ------------------
    - image_shape: tuple (H,W,C)
        H - image height
        W - image width
        C - number of channels
    - loader: Function
        Should take path to image file and image_shape as parameters.
        Should return numpy array
      
    Returns:
    --------
    - samples: ndarray (N, H, W, C)
        Numpy array of randomly chosen images resized according to model's input shape.
        N - number of samples
        H - height
        W - width
        C - number of channels
    - batch_labels: [int]
        Sample labels.
    """

    image_shape = kwargs.pop('image_shape', None)
    loader = kwargs.pop('loader', cv2_loader)

    dirs = np.random.choice(path_dirs, batch_size) if batch_size != None else path_dirs
    
    samples = np.zeros((num_per_class * len(dirs), *image_shape))
    batch_labels = np.ones(num_per_class * len(dirs)).astype(int)
    
    for i, dir_name in enumerate(dirs):
        dir_path = os.path.join(path, dir_name)
        filenames = os.listdir(dir_path)
        filenames = np.random.choice(filenames, num_per_class)
        
        batch = np.zeros((num_per_class, *image_shape))

        for j, filename in enumerate(filenames):
            batch[j,:,:,:] = loader(os.path.join(dir_path, filename), image_shape)
        
        samples[i*num_per_class: i*num_per_class + num_per_class, :, :, :] = batch
        batch_labels[i*num_per_class: i*num_per_class + num_per_class] = batch_labels[i*num_per_class: i*num_per_class + num_per_class] * labels[i]
    
    return samples, batch_labels

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