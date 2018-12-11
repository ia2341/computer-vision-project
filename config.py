import numpy as np

content_path = 'content/turtle.jpg'
style_path = 'style/picasso.jpg'

ARTWORK_FOLDER = "style"
PICTURE_FOLDER = "content"
OUTPUT_FOLDER = "output"

MAX_DIM = 512

# For de-processing processed images
CHANNEL_MEANS = np.array([103.939, 116.779, 123.68])

content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

UPDATE_EPOCH = 200
