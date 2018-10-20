from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import VGG16
import cv2
import numpy as np
orig = cv2.imread("fire.jpg")
xs = np.load()
ys = np.load()
model = VGG16(weights="imagenet")
model.fit(xs, ys)
image = cv2.resize(orig, (224, 224))
image = image_utils.img_to_array(image)
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)
preds = model.predict(image)
p = decode_predictions(preds)
print(p)