import numpy as np

ARTWORK_FOLDER = "artwork"
PICTURE_FOLDER = "pictures"
OUTPUT_FOLDER = "output"

MAX_DIM = 512

# For de-processing processed images
NORM_MEANS = np.array([103.939, 116.779, 123.68])

content_layers = ['block5_conv2']
style_layers = ['block1_conv1, block2_conv1, block3_conv1, block4_conv1, block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)
