import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.python.keras.preprocessing import image as kp_image

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
    
    return image


def deprocess_image(processed_image):

    image = processed_image.copy()
    if len(image.shape) == 4:
        image = np.squeeze(image, axis = 0)

    # This reverses the process of tf.keras.applications.vgg19.preprocess_input
    norm_grads = config.CHANNEL_MEANS
    image[:, :, 0] += norm_grads[0]
    image[:, :, 1] += norm_grads[1]
    image[:, :, 2] += norm_grads[2]
    image = image[:, :, ::-1]

    image = np.clip(image, 0, 255).astype('uint8')
    return image


def imshow(image):

    # Remove the batch dimension
    image = np.squeeze(image, axis = 0)
    plt.imshow(image.astype('uint8'))
