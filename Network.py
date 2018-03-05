import h5py
import os
import cv2
import tensorflow as tf
import tflearn
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_augmentation import ImageAugmentation

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

class Network:
	def Define():
		img_aug = ImageAugmentation()
		img_aug.add_random_flip_leftright()
		img_aug.add_random_crop((48, 48),6)
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
				
		return network

	def Train(h5_dataset,model_name,run_name,pre_load = False,tb_dir = './tfboard/'):
		h5f = h5py.File(h5_dataset, 'r')
		X = h5f['X'] #images
		Y = h5f['Y'] #labels
		X = np.reshape(X, (-1, 48, 48, 1))

		network = Network.Define()
		network = regression(network, optimizer='momentum',
				     loss='categorical_crossentropy')   

		model = tflearn.DNN(network,
				    max_checkpoints=1,
				    checkpoint_path="./Utils/", 
				    tensorboard_dir=tb_dir, 
				    tensorboard_verbose=3)
		
		if (pre_load == True):
			model.load(model_name)

		model.fit(X, Y, n_epoch=1, validation_set=0.15, shuffle=True,
			  show_metric=True, batch_size=100,
			  snapshot_epoch=True, run_id=run_name)

		model.save(model_name)

	def Test(output,model_name,test_dir,cascade_file):
		cascade_classifier = cv2.CascadeClassifier(cascade_file)

		network = Network.Define()
		model = tflearn.DNN(network)
		model.load(model_name)
		
		o = open(output,"a")
		c = 0
		l = os.listdir(test_dir)

		for f in l:
			img = cv2.imread(test_dir+'/'+f)
			result = model.predict(Network._FormatImage(img,cascade_classifier).reshape(1,48,48,1))

			labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
			dic = {}
			for i in range(0,7):
				dic[labels[i]] = result[0][i]

			sorted_dic = sorted(dic.items(), key=lambda kv: kv[1], reverse=True)
			first = (sorted_dic[0][0]).lower()
			if (first in f):
				c+=1

			o.write(f + ":\n" + str(sorted_dic) + "\n_____________\n")
	
		o.write("\n-- Total right: " + str(c) + " over " + str(len(l)))
		o.close()
	
	def TestBulk(model_name,test_dir,cascade_file,out_dir):
		cascade_classifier = cv2.CascadeClassifier(cascade_file)

		network = Network.Define()
		model = tflearn.DNN(network)
		model.load(model_name)

		for f in os.listdir(test_dir):
			img = cv2.imread(test_dir+'/'+f)
			print(test_dir+'/'+f)
			result = model.predict(Network._FormatImage(img,cascade_classifier).reshape(1,48,48,1))

			labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
			dic = {}
			for i in range(0,7):
				dic[labels[i]] = result[0][i]

			sorted_dic = sorted(dic.items(), key=lambda kv: kv[1], reverse=True)
			
			img = Image.open(test_dir+'/'+f)
			draw = ImageDraw.Draw(img)
			font = ImageFont.truetype("./Utils/micross.ttf", 16)
			draw.text((0, 0),str(sorted_dic),(255,255,255),font=font)
			draw.text((0, 30),str(sorted_dic),(0,0,0),font=font)
			img.save(out_dir+'/'+f)


	def _FormatImage(image,cascade_classifier):
		if len(image.shape) > 2 and image.shape[2] == 3:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		else:
			image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)

		faces = cascade_classifier.detectMultiScale(image,scaleFactor = 1.3,minNeighbors = 5)

		# None is we don't found any face - try to give back the whole picture anyway, but probably won't work welll
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
		except Exception: # Problem during resize
			return None
		
		return image

#Network.Train("./dataset/dataset.h5","./Model/model.tfl","nome_run",False,"./TFBoard/")
Network.Test("./Tests/TestResults/random.txt","./Model/model.tfl","./Tests/TestImages/","./Utils/h.xml")
#Network.TestBulk("./Model/model.tfl","./BulkTest/Input","./Utils/h.xml","./BulkTest/Results")
