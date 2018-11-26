import numpy as np

STYLE_FOLDER = "style"
CONTENT_FOLDER = "content"
OUTPUT_FOLDER = "output"

CONTENT_FILE = 'turtle.jpg'
STYLE_FILE = 'wave.jpg'

MAX_DIM = 512

# For de-processing processed images
CHANNEL_MEANS = np.array([103.939, 116.779, 123.68])

content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

UPDATE_EPOCH = 1
