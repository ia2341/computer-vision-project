import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import time
import functools
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

import config


def load_image(image_file):
    
    # Get the image and a scaling factor
    image = Image.open(image_file)
    image_scale = config.MAX_DIM / max(image.size)

    # Resize the image according to the scale and get it in array form
    image = image.resize((round(image.size[0] * image_scale), round(image.size[1] * image_scale)), Image.ANTIALIAS)
    image = kp_image.img_to_array(image)

    # Add a dimension in the 0th axis to account for the batch size
    image = np.expand_dims(image, axis = 0)
    
    return image.astype('uint8')


def process_image(image):
    image = tf.keras.applications.vgg19.preprocess_input(image)
    return image


def load_and_process_image(image_file):
    image = load_image(image_file).astype('float32')
    return process_image(image)


def deprocess_image(processed_image):

    image = processed_image.copy()
    if len(image.shape) == 4:
        image = np.squeeze(image, axis = 0)

    # This reverses the process of tf.keras.applications.vgg19.preprocess_input
    norm_grads = config.NORM_GRADS
    image[:, :, 0] += norm_grads[0]
    image[:, :, 1] += norm_grads[1]
    image[:, :, 2] += norm_grads[2]
    image = x[:, :, ::-1]

    image = np.clip(image, 0, 255).astype('uint8')
    return image


def imshow(image, caption=None):

    # Remove the batch dimension
    image = np.squeeze(image, axis = 0)

    # Show the image
    if caption is not None:
        plt.title(caption)
    plt.imshow(image.astype('uint8'))


def main():

    # set mpl parameters
