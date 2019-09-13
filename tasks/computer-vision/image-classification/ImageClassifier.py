import pandas as pd
import mlflow
import mlflow.keras
from keras.preprocessing.image import img_to_array, array_to_img
import numpy as np 
import os, time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras import models
from keras.models import Model
from keras import layers
from keras import optimizers
from keras import callbacks
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16;
from keras.applications.vgg16 import preprocess_input
import os
from sklearn.metrics import classification_report


if __name__ == '__main__':
	mlflow.keras.autolog()

	os.getcwd()
	os.listdir(os.getcwd())

	df = pd.read_csv("fashion-mnist_train.csv") 
	df1 =pd.read_csv("fashion-mnist_test.csv")
		
	print(df.shape)
	print(df1.shape)
	
	df.shape 
	df1.shape 
	train_X= np.array(df.iloc[:,1:])
	test_X= np.array(df1.iloc[:,1:])
	train_Y= np.array (df.iloc[:,0]) 
	test_Y = np.array(df1.iloc[:,0])

	train_X.shape, test_X.shape

	classes = np.unique(train_Y)
	num_classes = len(classes)
	num_classes

	# Images are converted into 3 channels and reshape them
	train_X=np.dstack([train_X] * 3)
	test_X=np.dstack([test_X]*3)
	
	train_X = train_X.reshape(-1, 28,28,3)
	test_X= test_X.reshape (-1,28,28,3)
	train_X.shape,test_X.shape
	
	# Images are resized to 48*48 for VGG16 model
	train_X = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in train_X])
	test_X = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in test_X])
	
	# data normalization and data type is changed
	train_X = train_X / 255.
	test_X = test_X / 255.
	train_X = train_X.astype('float32')
	test_X = test_X.astype('float32')
	
	# Labels are converted to one hot encoded format, train data is split into train and validation data 
	train_Y_one_hot = to_categorical(train_Y)
	test_Y_one_hot = to_categorical(test_Y)
	train_X,valid_X,train_label,valid_label = train_test_split(train_X,train_Y_one_hot,test_size=0.2,random_state=13)
	
	train_X.shape,valid_X.shape,train_label.shape,valid_label.shape
	
	#Paramters for VGG16 model
	Image_Width=48
	Image_Height=48
	Image_Depth=3
	Batch_Size=16
	
	# Input is preprocessed 
	train_X = preprocess_input(train_X)
	valid_X = preprocess_input(valid_X)
	test_X  = preprocess_input (test_X)
	
	#  Create base model of VGG16
	conv_base = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',include_top=False,input_shape=(Image_Height, Image_Width, Image_Depth))
	conv_base.summary()

	# Extracting features
	train_features = conv_base.predict(np.array(train_X), batch_size=Batch_Size, verbose=1)
	test_features = conv_base.predict(np.array(test_X), batch_size=Batch_Size, verbose=1)
	val_features = conv_base.predict(np.array(valid_X), batch_size=Batch_Size, verbose=1)

	# Saving the features so that they can be used for future
	np.savez("train_features", train_features, train_label)
	np.savez("test_features", test_features, test_Y)
	np.savez("val_features", val_features, valid_label)

	# Current shape of features
	print(train_features.shape, "\n",  test_features.shape, "\n", val_features.shape)

	# Flatten extracted features
	train_features_flat = np.reshape(train_features, (48000, 1*1*512))
	test_features_flat = np.reshape(test_features, (10000, 1*1*512))
	val_features_flat = np.reshape(val_features, (12000, 1*1*512))

	# Densely connected classifier followed by leakyrelu layer and finally dense layer for the number of classes
	NB_TRAIN_SAMPLES = train_features_flat.shape[0]
	NB_VALIDATION_SAMPLES = val_features_flat.shape[0]
	NB_EPOCHS = 100

	keras_model = models.Sequential()
	keras_model.add(layers.Dense(512, activation='relu', input_dim=(1*1*512)))
	keras_model.add(layers.LeakyReLU(alpha=0.1))
	keras_model.add(layers.Dense(num_classes, activation='softmax'))

	# Model Compilation
	keras_model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(), metrics=['acc'])

	# reduced learning and the early stopping for callback methos
	reduce_learning = callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=2,verbose=1,mode='auto',epsilon=0.0001,cooldown=2,min_lr=0)

	eary_stopping = callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=7,verbose=1,mode='auto')

	callbacks = [reduce_learning, eary_stopping]

	# Model Training
	history =keras_model.fit(train_features_flat,train_label,epochs=NB_EPOCHS,validation_data=(val_features_flat, valid_label),callbacks=callbacks)

	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1, len(acc) + 1)

	plt.title('Training and validation accuracy')
	plt.plot(epochs, acc, 'red', label='Training acc')
	plt.plot(epochs, val_acc, 'blue', label='Validation acc')
	plt.legend()

	plt.figure()
	plt.title('Training and validation loss')
	plt.plot(epochs, loss, 'red', label='Training loss')
	plt.plot(epochs, val_loss, 'blue', label='Validation loss')

	plt.legend()
	plt.show()

	# get the predictions for the test data
	predicted_classes = keras_model.predict_classes(test_features_flat)



	# get the indices to be plotted
	y_true = df1.iloc[:, 0]
	correct = np.nonzero(predicted_classes==y_true)[0]
	incorrect = np.nonzero(predicted_classes!=y_true)[0]

	target_names = ["Class {}".format(i) for i in range(10)]
	report= classification_report(y_true, predicted_classes, target_names=target_names)

	print(report)

	"""The model lacks precision for class 6, and underperforms for class 0 in reference fo rthe rest of them."""

	mlflow.log_param('Image_Width', Image_Width)
	mlflow.log_param('Image_Height', Image_Height)
	mlflow.log_param('Image_Depth',Image_Depth)
#	mlflow.log_metric('report', report)




