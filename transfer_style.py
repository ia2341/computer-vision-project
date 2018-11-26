# import the necessary modules
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import tensorflow.contrib.eager as tfe

import config
from processing import load_image, deprocess_image
import model


def loss_and_gradients(my_model, output_image, content_features, gram_style_features, alpha = 10000, beta = 1):
    ''' Function that computes the gradients for the output image based on the actual content and style
    features as well as their respective multipliers. '''
    with tf.GradientTape() as g:
        loss_ = model.loss(my_model, output_image, content_features, gram_style_features, alpha, beta)
    return loss_, g.gradient(loss_, output_image)


def transfer_style(content_image, style_image, epochs = 10000, alpha = 10000, beta = 1):
    ''' Function that transfers the style from the content_image to the style_image. 
    The gradient descent algorithm is performed over 'epochs' iterations and the 
    multiplier for the content and style are specified by alpha and beta respectively. '''

    # get the model
    my_model = model.model()
    for layer in my_model.layers:
        layer.trainable = False

    # get the actual content and style representations
    content_features, style_features = model.feature_representations(my_model, content_image, style_image)
    gram_style_features = [model.gram_matrix(feature) for feature in style_features]

    # create a random noise image
    output_image = np.random.randint(0, 256, size = content_image.shape).astype('float32')
    output_image = tf.keras.applications.vgg19.preprocess_input(output_image)
    output_image = tfe.Variable(output_image, dtype=tf.float32)

    # create tracking variables and the Adam optimizer
    counter, min_loss, best_image = 1, float('inf'), None
    optimizer = tf.train.AdamOptimizer(5, beta1=0.99, epsilon=0.1)

    # set the minimum and maximum values allowed in the image accounting for channel means
    minimum, maximum = - config.CHANNEL_MEANS, 255 - config.CHANNEL_MEANS

    # start the timer
    start_time = time.time()
    update_start_time = time.time()

    for i in range(epochs):

        # get the loss and gradients for the output image based on the current iteration
        loss_, curr_gradients = loss_and_gradients(my_model, output_image, content_features, gram_style_features, alpha, beta)

        # apply the gradients based on the current loss
        optimizer.apply_gradients([(curr_gradients, output_image)])

        # clip the new matrix based on the minimum and maximum values
        output_image.assign(tf.clip_by_value(output_image, minimum, maximum))

        # if a new best image is found
        if loss_ < min_loss:
            min_loss = loss_
            best_image = output_image.numpy()

        # print loss and time statistics after every UPDATE_EPOCH iterations
        if i % config.UPDATE_EPOCH == 0:

            print("Iteration: {} \t Total Loss: {}".format(i, loss_))
            print("Time for {} epochs = {} seconds.".format(config.UPDATE_EPOCH, time.time() - update_start_time))
            update_start_time = time.time()

    # print the total time taken to perform the optimization algorithm
    print("\nTotal Time: {} seconds".format(time.time() - start_time))

    # return the best found image and the associated overall loss
    return deprocess_image(best_image), min_loss


def main():
    ''' Main method that performs the style transfer algorithm '''

    # get the file paths for the content and style images
    content_path = config.CONTENT_FOLDER + '/' + config.CONTENT_FILE
    style_path = config.STYLE_FOLDER + '/' + config.STYLE_FILE

    # load the desired images
    print('Loading the input image...')
    content_image = load_image(content_path)
    style_image = load_image(style_path)

    # process the images
    content_image = tf.keras.applications.vgg19.preprocess_input(content_image).astype('float32')
    style_image = tf.keras.applications.vgg19.preprocess_input(style_image).astype('float32')

    # transfer the style of style_image onto content_image and save to file
    output_image, mis_loss = transfer_style(content_image, style_image)
    im = Image.fromarray(output_image)
    im.save(config.OUTPUT_FOLDER + '/' + 'output-' + time.time() + '.jpg')


# if the main method is called
if __name__ == '__main__':

    # enable eager execution
    print('Enabling Eager Execution...')
    tf.enable_eager_execution()

    # call the main method
    main()
