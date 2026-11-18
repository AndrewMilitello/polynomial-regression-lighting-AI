import numpy as np
np.random.seed(1)
hidden_layer_1_size = 512
hidden_layer_2_size = 128
output_degree = 5
bsize = 10
sample_size = 5 # size of patches sample from image
training_size = 4000
LEARNING_RATE = 0.0005
EPOCHS = 200
ERROR_THRESHOLD = 0.002
MAX_DIM = 500 # maximum dimension of input images, images will be resized to fit this
train_amplitude = 2 # scale image strength by this factor