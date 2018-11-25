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
from processing import load_and_process_image, process_image, deprocess_image
import model


def gradients(my_model, output_image, content_features, gram_style_features, alpha = 5e-3, beta = 1):
    with tf.GradientTape() as g:
        loss, content_loss, style_loss = model.loss(my_model, output_image, content_features, gram_style_features, alpha, beta)
    return g.gradient(loss, output_image)


def transfer_style(content_image, style_image, epochs = 1000, alpha = 1000, beta = 0.01):

    my_model = model.model()
    for layer in my_model.layers:
        layer.trainable = False

    content_features, style_features = model.feature_representations(my_model, content_image, style_image)
    gram_style_features = [model.gram_matrix(feature) for feature in style_features]

    output_image = np.random.randint(0, 256, size = content_image.shape).astype('uint8')
    output_image = process_image(output_image)
    output_image = tfe.Variable(output_image, dtype=tf.float32)

    counter, min_loss, best_image = 1, float('inf'), None
    optimizer = tf.train.AdamOptimizer(5, beta1=0.99, epsilon=0.1)

    minimum, maximum = - config.CHANNEL_MEANS, 255 - config.CHANNEL_MEANS

    start_time = time.time()
    update_start_time = time.time()

    for i in range(epochs):

        loss, content_loss, style_loss = model.loss(my_model, output_image, content_features, gram_style_features, alpha, beta)
        curr_gradients = gradients(my_model, output_image, content_features, gram_style_features, alpha, beta)
        optimizer.apply_gradients([(curr_gradients, output_image)])
        
        output_image.assign(tf.clip_by_value(output_image, minimum, maximum))

        if loss < min_loss:
            min_loss = loss
            best_image = output_image.numpy()

        if counter % config.UPDATE_EPOCH == 0:
            print("Iteration: {} \t Total Loss: {} \t Content Loss: {} \t Style Loss: {}".format(i, loss, content_loss, style_loss))
            print("Time for {} epochs = {} seconds.".format(config.UPDATE_EPOCH, time.time() - update_start_time))
            update_start_time = time.time()

    print()
    print("Total Time: {} seconds".format(time.time() - start_time))

    return deprocess_image(best_image), min_loss


def main():

    content_path = 'pictures/turtle.jpg'
    style_path = 'artwork/wave.jpg'

    print('Loading the input image...')
    content_image = load_and_process_image(content_path)
    style_image = load_and_process_image(style_path)

    output_image, mis_loss = transfer_style(content_image, style_image)
    im = Image.fromarray(output_image)
    im.save(config.OUTPUT_FOLDER + '/' + 'output-' + time.time() + '.jpg')


if __name__ == '__main__':

    print('Enabling Eager Execution...')
    tf.enable_eager_execution()

    main()
