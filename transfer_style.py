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
from preprocessing import load_and_process_image, process_image
import model



def transfer_style(content_image, style_image, epochs = 1000, alpha = 1000, beta = 0.01):

    model = model.model()
    for layer in model.layers:
        layer.trainable = False

    content_features, style_features = model.feature_representations(model, content_image, style_image)
    gram_style_features = [model.gram_matrix(feature) for feature in style_features]

    output_image = np.random.randint(0, 256, size = content_image.shape).astype('uint8')
    output_image = process_image(output_image)
    output_image = tfe.Variable(output_image, dtype=tf.float32)

    counter, min_loss, best_image = 1, float('inf'), None



def main():

    content_path = 'pictures/turtle.jpg'
    style_path = 'artwork/wave.jpg'

    print('Loading the input image...')
    content_image = load_and_process_image(content_path)
    style_image = load_and_process_image(style_path)

    output_image = transfer_style(content_image, style_image)


if __name__ == '__main__':

    print('Enabling Eager Execution...')
    tf.enable_eager_execution()

    main()
