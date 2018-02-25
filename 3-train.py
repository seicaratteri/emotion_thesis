import h5py
import tensorflow as tf
import tflearn
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

h5f = h5py.File('dataset.h5', 'r')
X = h5f['X'] #images
Y = h5f['Y'] #labels
X = np.reshape(X, (-1, 48, 48, 1))

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_crop((24, 24))
img_aug.add_random_rotation(max_angle=25.)

network = input_data(shape=[None, 48, 48, 1], data_augmentation=img_aug) #48 x 48 grayscale
network = conv_2d(network, 64, 5, activation='relu')
#network = local_response_normalization(network)
network = max_pool_2d(network, 3, strides=2)
network = conv_2d(network, 64, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = conv_2d(network, 128, 4, activation='relu')
network = dropout(network,0.3)
network = fully_connected(network, 3072, activation='tanh')
network = fully_connected(network, 7, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy')   

# Training
model = tflearn.DNN(network, checkpoint_path='check/',
                    max_checkpoints=1, tensorboard_dir='./tfboard/', tensorboard_verbose=3)

#model.load("model.tfl")

model.fit(X, Y, n_epoch=80, validation_set=0.15, shuffle=True,
          show_metric=True, batch_size=100,
          snapshot_epoch=True, run_id='test_augmentation')

model.save("model.tfl")
