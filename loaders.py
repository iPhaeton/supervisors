import tensorflow as tf
from PIL import Image
import numpy as np
from tqdm import tqdm
import os

def load_batch_of_images(path, dirs, labels, num_per_class, image_shape):
    """
    Loads a random batch of images
    
    Parameters:
    -----------
    - path: string
        Path to the image source directory on disk. 
        Source directory should be divided into directories, one directory per class.
    - dirs: [string]
        List of directories contained in the path (classes).
    - labels: [int]
        Class labels. Should correspond to classes.
    - num_per_class: int
        Number of images randomly chosen from each class
    - image_shape: tuple (H,W,C)
        H - image height
        W - image width
        C - number of channels
      
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
    
    samples = np.zeros((num_per_class * len(dirs), *image_shape))
    batch_labels = np.ones(num_per_class * len(dirs)).astype(int)
    
    for i, dir_name in enumerate(tqdm(dirs)):
        dir_path = os.path.join(path, dir_name)
        filenames = os.listdir(dir_path)
        filenames = np.random.choice(filenames, num_per_class)
        
        batch = np.zeros((num_per_class, *image_shape))

        for j, filename in enumerate(filenames):
            img = Image.open(os.path.join(dir_path, filename))
            img = img.resize((image_shape[1], image_shape[0]))
            img = np.array(img)
            batch[j,:,:,:] = img
        
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