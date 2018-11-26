import numpy as np
import tensorflow as tf
from PIL import Image
import time
import tensorflow.contrib.eager as tfe

import config
from processing import load_image, deprocess_image
import model


def gradients(my_model, output_image, content_features, gram_style_features, alpha = 10000, beta = 1):
    with tf.GradientTape() as g:
        loss_ = model.loss(my_model, output_image, content_features, gram_style_features, alpha, beta)
    return g.gradient(loss_, output_image)

def transfer_style(content_image, style_image, epochs = 10000, alpha = 10000, beta = 1):

    my_model = model.model()
    for layer in my_model.layers:
        layer.trainable = False

    content_features, style_features = model.feature_representations(my_model, content_image, style_image)
    gram_style_features = [model.gram_matrix(feature) for feature in style_features]

    output_image = np.random.randint(0, 256, size = content_image.shape).astype('float32')
    output_image = tf.keras.applications.vgg19.preprocess_input(output_image)
    output_image = tfe.Variable(output_image, dtype=tf.float32)

    counter, min_loss, best_image = 1, float('inf'), None
    optimizer = tf.train.AdamOptimizer(5, beta1=0.99, epsilon=0.1)

    minimum, maximum = - config.CHANNEL_MEANS, 255 - config.CHANNEL_MEANS

    start_time = time.time()
    update_start_time = time.time()

    for i in range(epochs):

        loss_ = model.loss(my_model, output_image, content_features, gram_style_features, alpha, beta)
        curr_gradients = gradients(my_model, output_image, content_features, gram_style_features, alpha, beta)
        optimizer.apply_gradients([(curr_gradients, output_image)])

        output_image.assign(tf.clip_by_value(output_image, minimum, maximum))

        if loss_ < min_loss:
            min_loss = loss_
            best_image = output_image.numpy()

        if i % config.UPDATE_EPOCH == 0:

            print("Iteration: {} \t Total Loss: {}".format(i, loss_))
            print("Time for {} epochs = {} seconds.".format(config.UPDATE_EPOCH, time.time() - update_start_time))
            update_start_time = time.time()

    print()
    print("Total Time: {} seconds".format(time.time() - start_time))

    return deprocess_image(best_image), min_loss


def main():

    content_path = config.CONTENT_FOLDER + '/' + config.CONTENT_FILE
    style_path = config.STYLE_FOLDER + '/' + config.STYLE_FOLDER

    print('Loading the input image...')
    content_image = load_image(content_path)
    style_image = load_image(style_path)

    content_image = tf.keras.applications.vgg19.preprocess_input(content_image).astype('float32')
    style_image = tf.keras.applications.vgg19.preprocess_input(style_image).astype('float32')

    output_image, mis_loss = transfer_style(content_image, style_image)
    im = Image.fromarray(output_image)
    im.save(config.OUTPUT_FOLDER + '/' + 'output-' + time.time() + '.jpg')


if __name__ == '__main__':

    print('Enabling Eager Execution...')
    tf.enable_eager_execution()

    main()
