################################################################################
#
# We would like to acknowledge a source that helped us process and deprocess
# images for the purpose of this assignment. This includes using eager execution
# in tensorflow, loading, preparing, and deprocessing and saving the images as
# they traverse through the neural network and the custom optimizer.
#
# Acknowledgements:
# https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398
#
################################################################################


# import the necessary modules
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.keras.preprocessing import image as pre

import config


def load_image(image_file):
    ''' Function that loads an image from an input file path.
    The function will return a numpy array that represents the
    pixel values of the image.'''
    
    # Get the image and a scaling factor
    image = Image.open(image_file)
    image_scale = config.MAX_DIM / max(image.size)

    # Resize the image according to the scale and get it in array form
    image = image.resize((round(image.size[0] * image_scale), round(image.size[1] * image_scale)), Image.ANTIALIAS)
    image = pre.img_to_array(image)

    # Add a dimension in the 0th axis to account for the batch size
    image = np.expand_dims(image, axis = 0)
    
    return image


def deprocess_image(processed_image):
    ''' Function that de-processes the image after it finishes
    a forward pass of the neural model. This processing includes
    shifting the pixel values by the normalized mean and
    removing the "batch" dimension. '''

    # remove the "batch" dimension from the image
    image = processed_image.copy()
    if len(image.shape) == 4:
        image = np.squeeze(image, axis = 0)

    # This reverses the process of tf.keras.applications.vgg19.preprocess_input
    image[:, :, 0] += config.CHANNEL_MEANS[0]
    image[:, :, 1] += config.CHANNEL_MEANS[1]
    image[:, :, 2] += config.CHANNEL_MEANS[2]
    image = image[:, :, ::-1]

    # ensure pixel values are within 
    image = np.clip(image, 0, 255).astype('uint8')
    return image
