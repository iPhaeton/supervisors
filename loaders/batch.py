from decorators import partially_applied
import numpy as np
import os
from PIL import Image
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
def load_batch_of_images(path, path_dirs, labels, num_per_class, image_shape, loader, batch_size=None):
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

def load_batch_of_data(samples, labels, batch_size, iteration):
    start_idx = iteration * batch_size
    end_idx = start_idx + batch_size
    batch_indices = range(samples.shape[0])[start_idx : end_idx]

    batch = samples[batch_indices,:,:,:]
    batch_labels = labels[batch_indices]
    return batch, batch_labels
