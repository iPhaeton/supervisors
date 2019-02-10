import tensorflow as tf
from PIL import Image
import numpy as np
import os
from pyramda import curry
from decorators import partially_applied
import cv2
import platform
from six.moves import cPickle as pickle

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

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(path):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,4):
        f = os.path.join(path, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(path, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

def load_CIFAR10_data(path, num_training=29000, num_validation=1000, num_test=10000):
    # Original code: 
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = path
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test
