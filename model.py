import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import time
import functools
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

import config


def model():

    original_vgg = tf.keras.applications.vgg19.VGG19(include_top = False, weights = 'imagenet')
    vgg.trainable = False

    content_outputs = [vgg.get_layer(layer) for layer in config.content_layers]
    style_outputs = [vgg.get_layer(layer) for layer in config.style_layers]

    outputs = style_outputs + content_outputs
    return models.Model(vgg.input, outputs)


def gram_matrix():
    pass

def style_loss():
    pass


def content_loss(predicted, actual):
    ''' Based on the implementation described in 
    https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
    '''
    return 0.5 * tf.reduce_sum(tf.square(predicted - actual))
