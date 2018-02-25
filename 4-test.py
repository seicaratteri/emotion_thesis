import h5py
import cv2
import tensorflow as tf
import numpy as np
from numpy import array
import tflearn
from PIL import Image
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


cascade_classifier = cv2.CascadeClassifier("./h.xml")

def format_image(image):
	if len(image.shape) > 2 and image.shape[2] == 3:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	else:
		image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)

	faces = cascade_classifier.detectMultiScale(image,scaleFactor = 1.3,minNeighbors = 5)

	# None is we don't found an image
	if not len(faces) > 0:
		return cv2.resize(image, (48, 48), interpolation = cv2.INTER_CUBIC) / 255.
		#return None
	max_area_face = faces[0]
	for face in faces:
		if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
			max_area_face = face
	# Chop image to face
	face = max_area_face
	image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
	# Resize image to network size
	try:
		image = cv2.resize(image, (48, 48), interpolation = cv2.INTER_CUBIC) / 255.
	except Exception:
		print("[+] Problem during resize")
		return None
	
	return image

network = input_data(shape=[None, 48, 48, 1]) #48 x 48 grayscale
network = conv_2d(network, 64, 5, activation='relu')
#network = local_response_normalization(network)
network = max_pool_2d(network, 3, strides=2)
network = conv_2d(network, 64, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = conv_2d(network, 128, 4, activation='relu')
network = dropout(network,0.3)
network = fully_connected(network, 3072, activation='tanh')
network = fully_connected(network, 7, activation='softmax')

model = tflearn.DNN(network)
model.load("model.tfl")

img = cv2.imread("./testimage/sad1.png")
result = model.predict(format_image(img).reshape(1,48,48,1))

print("\n")
labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
dic = {}
for i in range(0,7):
	dic[labels[i]] = result[0][i]

sorted_dic = sorted(dic.items(), key=lambda kv: kv[1], reverse=True)
print(sorted_dic)
print("\n")
#print(result)
#print("(" + labels[result[0][0]] + ")")
