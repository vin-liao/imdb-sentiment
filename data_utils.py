import numpy as np
from keras.datasets import imdb
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence

max_len = 1500 #completely arbitrary
#the outlier of the data contains around 2.7k words, I'm not gonna use it.

def load_data(size=0.2):
	# load the dataset
	(X_train, y_train), (X_test, y_test) = imdb.load_data()
	X = np.concatenate((X_train, X_test), axis=0)
	y = np.concatenate((y_train, y_test), axis=0)
	vocab_size = len(imdb.get_word_index())

	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)

	x_train = sequence.pad_sequences(x_train, maxlen=max_len)
	x_test = sequence.pad_sequences(x_test, maxlen=max_len)

	return x_train, x_test, y_train, y_test

def get_wi():
	return imdb.get_word_index()

def get_max_len():
	return max_len