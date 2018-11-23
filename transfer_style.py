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

from preprocessing import load_image


def main():

    content_path = 'pictures/turtle.jpg'
    style_path = 'artwork/wave.jpg'

    print('Loading the input image...')
    content_image = load_image(content_path)
    style_image = load_image(style_path)


if __name__ == '__main__':

    print('Setting MatPlotLib params...')
    mpl.rcParams['figure.figSize'] = (10, 10)
    mpl.rcParams['axes.grid'] = False

    print('Enabling Eager Execution...')
    tf.enable_eager_execution()

    main()
