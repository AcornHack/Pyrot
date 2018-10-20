import cv2
import os
import sys
import math

import tflearn
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression
import cv2 as cv
def construct_firenet (x,y):
    # Build network as per architecture in [Dunnings/Breckon, 2018]
    network = tflearn.input_data(shape=[None, y, x, 3], dtype=tf.float32)
    network = conv_2d(network, 64, 5, strides=4, activation="relu")
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 128, 4, activation="relu")
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 1, activation="relu")
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation="tanh")
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation="tanh")
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation="softmax")
    network = regression(network, optimizer="momentum",
                         loss="categorical_crossentropy",
                         learning_rate=0.001)
    model = tflearn.DNN(network, checkpoint_path="firenet",
                        max_checkpoints=1, tensorboard_verbose=2)
    return model


rows = 224
cols = 224

model = construct_firenet(rows, cols)
model.load(os.path.join("models/FireNet", "firenet"), weights_only=True)
print("\n---------------------\n")

def prediction(img):
    im = cv.resize(cv.imread(img), (rows, cols), cv.INTER_AREA)
    output = model.predict([im])
    return (["Fire", "No Fire"][np.argmax(output)], np.max(output))

if len(sys.argv) == 2:
    print("Prediction: {}".format(prediction(sys.argv[1])))
else:
    print("Prediction for fire.jpg is {}".format(prediction("fire.jpg")))





