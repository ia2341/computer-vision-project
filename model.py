import tensorflow as tf
from tensorflow.python.keras import models

import config


def model():

    original_vgg = tf.keras.applications.vgg19.VGG19(include_top = False, weights = 'imagenet')

    content_outputs = [original_vgg.get_layer(layer).output for layer in config.content_layers]
    style_outputs = [original_vgg.get_layer(layer).output for layer in config.style_layers]

    return models.Model(original_vgg.input, content_outputs + style_outputs)


def gram_matrix(feature):
    ''' Based on the implementation described in
    https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
    '''
    # flatten the features to one row per channel
    num_channels = feature.shape[-1]
    feature = tf.reshape(feature, (-1, num_channels))

    return tf.matmul(feature, feature, transpose_a = True)


def gram_style_loss(predicted_gram, actual_gram, pixels, channels):
    ''' Based on the implementation described in
    https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
    '''
    loss_ = tf.reduce_sum(tf.square(predicted_gram - actual_gram))
    return loss_ / (4 * (pixels ** 2) * (channels ** 2))


def content_loss(predicted, actual):
    ''' Based on the implementation described in
    https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
    '''
    return 0.5 * tf.reduce_sum(tf.square(predicted - actual))


def loss(model, output_image, actual_content_features, actual_gram_features, alpha = 10000, beta = 1):

    model_representation = model(output_image)
    content_representation = model_representation[ : len(config.content_layers)]
    style_representation = model_representation[len(config.content_layers) : ]

    total_content_loss, total_style_loss = 0, 0

    for actual_content, predicted_content in zip(actual_content_features, content_representation):
        total_content_loss += content_loss(predicted_content, actual_content)

    for actual_gram, predicted_style in zip(actual_gram_features, style_representation):
        predicted_gram = gram_matrix(predicted_style)
        shape = tf.shape(predicted_style)
        total_style_loss += gram_style_loss(predicted_gram, actual_gram, float(shape[0] * shape[1]), float(shape[-1]))

    # normalize loss per layer
    total_content_loss /= len(config.content_layers)
    total_style_loss /= len(config.style_layers)

    loss_ = (alpha * total_content_loss) + (beta * total_style_loss)
    return loss_


def feature_representations(model, content_image, style_image):

    content = model(content_image)
    style = model(style_image)

    content_features = [layer for layer in content[ : len(config.content_layers)]]
    style_features = [layer for layer in style[len(config.content_layers) : ]]

    return content_features, style_features
