from flask import Flask, render_template, request, jsonify, redirect
from flask_cors import CORS
import cv2 as cv
import os

import os
import sys
import math

import tflearn
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression
import cv2 as cv

def construct_firenet(x,y):
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
    network = regression(network, optimizer="momentum", loss="categorical_crossentropy", learning_rate=0.001)
    model = tflearn.DNN(network, checkpoint_path="firenet", max_checkpoints=1, tensorboard_verbose=2)
    return model


rows = 224
cols = 224
model = construct_firenet(rows, cols)
model.load(os.path.join("models/FireNet", "firenet"), weights_only=True)

app = Flask(__name__, static_url_path="/static")
CORS(app)

@app.route("/")
def main():
	return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
	if request.method == "POST":
		f = request.files["file"]
		f.save(os.path.join("ims", request.files["file"].filename))

		im = cv.imread(os.path.join("ims", request.files["file"].filename))
		small = cv.resize(im, (rows, cols), cv.INTER_AREA)
		output = model.predict([small])
		re = (["Fire", "Clear"][np.argmax(output)], np.max(output), "large")
		if re[0] == "Fire":
			return "I am {}% sure this is a fire. Size: {}. Evacuate immediately".format(math.trunc(re[1]*100), re[2])
		if re[1] == "Clear":
			return "This picture appears clear."
if __name__ == "__main__":
	app.run(host="127.0.0.1", port=3000, debug=True)
