import numpy as np
from scipy.misc import imresize

def rgb2gray(image):
    # convert RGB image into grayscale image.
    # gray = 0.299 * R + 0.587 * G + 0.114 * B
    return np.dot(image[...,:3], [0.299, 0.587, 0.114])

def transform_rgb2gray(image, shape):
    return imresize(rgb2gray(image)/255., shape)