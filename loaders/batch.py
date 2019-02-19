from decorators import partially_applied
import numpy as np
import os
from PIL import Image
import cv2
import math

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

def load_batch_of_images(path, dirs, labels, image_shape, loader, num_per_class, batch_size=None, random=False, indices=None):
    """
    Loads a random batch of images
    
    Keyword Arguments:
    -----------
    - path: string
        Path to the image source directory on disk. 
        Source directory should be divided into directories, one directory per class.
    - dirs: [string]
        List of directories to be considered in the path. Every directory should contain images of a single class.
    - labels: [int]
        Class labels. Should correspond to classes.
    - image_shape: tuple (H,W,C)
        H - image height
        W - image width
        C - number of channels
    - loader: Function
        Should take path to image file and image_shape as parameters.
        Should return numpy array
    - num_per_class: int
        Number of images that should be randomly chosen from each class.
    - batch_size: int
        Number of classes to use in a single batch. Total number of samples will be batch_size * num_per_class.
        If batch_size == None, samples will be taken from every one of dirs.
    - random: bool
        If classes for a batch should be chosen randomly.
    - start_idx: int
        Start index, if batch is not random.
    - end_idx: int
        End index, if batch is not random.
      
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

    # Either random = True or batch_size is specified or start_idx and end_idx are specified
    dirs = np.array(dirs)
    
    if batch_size == None:
        batch_dirs = dirs
    elif random == True:
        assert batch_size != None
        batch_dirs = np.random.choice(dirs, batch_size)
    else:
        assert (batch_size != None) & (indices is not None)
        batch_dirs = dirs[indices]

    samples = np.zeros((num_per_class * len(batch_dirs), *image_shape))
    batch_labels = np.ones(num_per_class * len(batch_dirs)).astype(int)
        
    for i, dir_name in enumerate(batch_dirs):
        dir_path = os.path.join(path, dir_name)
        filenames = os.listdir(dir_path)
        filenames = np.random.choice(filenames, num_per_class)
        #filenames = filenames[0:num_per_class]
            
        batch = np.zeros((num_per_class, *image_shape))

        for j, filename in enumerate(filenames):
            batch[j,:,:,:] = loader(os.path.join(dir_path, filename), image_shape)
            
        samples[i*num_per_class: i*num_per_class + num_per_class, :, :, :] = batch
        batch_labels[i*num_per_class: i*num_per_class + num_per_class] = batch_labels[i*num_per_class: i*num_per_class + num_per_class] * labels[i]
    
    return samples, batch_labels

def batch_of_images_generator(shuffle=True, **kwargs):
    """
    Generates batches of images
    
    Keyword Arguments:
    -----------
    - path: string
        Path to the image source directory on disk. 
        Source directory should be divided into directories, one directory per class.
    - dirs: [string]
        List of directories to be considered in the path. Every directory should contain images of a single class.
    - labels: [int]
        Class labels. Should correspond to classes.
    - image_shape: tuple (H,W,C)
        H - image height
        W - image width
        C - number of channels
    - loader: Function
        Should take path to image file and image_shape as parameters.
        Should return numpy array
    - num_per_class: int
        Number of images that should be randomly chosen from each class.
    - batch_size: int
        Number of classes to use in a single batch. Total number of samples will be batch_size * num_per_class
      
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

    dirs = kwargs.get('dirs')
    batch_size = kwargs.get('batch_size')
    if shuffle == True:
        indices = np.random.permutation(len(dirs))
    else:
        indices = range(len(dirs))

    iterations_per_epoch = math.ceil(len(dirs) / batch_size)

    iter_count = -1
    while True:
        iter_count += 1

        start_idx = iter_count * batch_size
        end_idx = min(start_idx + batch_size, len(dirs))

        samples, batch_labels = load_batch_of_images(**kwargs, random=False, indices=indices[start_idx:end_idx])
        
        if end_idx >= len(dirs):
            iter_count = -1
            if shuffle == True:
                indices = np.random.permutation(len(dirs))
        
        yield iter_count, samples, batch_labels

def load_batch_of_data(samples, labels, batch_size, iteration):
    start_idx = iteration * batch_size
    end_idx = start_idx + batch_size
    batch_indices = range(samples.shape[0])[start_idx : end_idx]

    batch = samples[batch_indices,:,:,:]
    batch_labels = labels[batch_indices]
    return batch, batch_labels
