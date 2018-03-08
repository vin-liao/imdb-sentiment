import numpy as np
import os
from requests import get
import zipfile
import data_utils

dim_size = 50
embedding_path = './data/glove.6B.50d.txt.tar.gz'
txt_path = './data/glove.6B.{}d.txt'.format(dim_size)
word_index = data_utils.get_wi()
vocab_size = len(word_index)

def download_embedding():
	#if txt file exist
	if os.path.isfile(txt_path) == False:
		zip_ref = zipfile.ZipFile(embedding_path, 'r')
		zip_ref.extractall('./data/')

def load_embedding():
	download_embedding()

	#load embedding, put it into a dictionary
	embedding_index = dict()
	f = open(txt_path)
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embedding_index[word] = coefs
	f.close()

	#mapping word to the coreesponding vectors
	embedding_matrix = np.zeros((len(word_index), dim_size))
	for word, i in word_index.items():
		embedding_vector = embedding_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

	return embedding_matrix

def get_dim():
	return dim_size