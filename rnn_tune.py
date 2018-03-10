import numpy as np
import embedding_utils
import data_utils
from keras.models import Sequential
import keras.optimizers
from keras import regularizers
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, CuDNNGRU, Dropout, BatchNormalization, Activation
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperopt import Trials, STATUS_OK, tpe

# x_train, x_test, y_train, y_test = data_utils.load_data()
word_index = data_utils.get_wi()
vocab_size = len(word_index)
embedding_matrix = embedding_utils.load_embedding()
dim_size = embedding_utils.get_dim()
max_len = data_utils.get_max_len()

def data():
	data_utils.load_data()

def create_model(x_train, x_test, y_train, y_test):
	model = Sequential()
	model.add(Embedding(vocab_size, dim_size, input_length=max_len, weights=[embedding_matrix], trainable=False))

	model.add(CuDNNGRU({{choice([32, 64, 128])}}))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout({{uniform(0, 1)}}))

	model.add(Dense(1, activation='sigmoid'))

	adam = keras.optimizers.Adam(lr={{choice([0.01, 0.001, 0.0001])}}, clipvalue=1000)
	model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
	print(model.summary())

	model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=200, batch_size={{choice([16, 32, 64, 128])}}, verbose=2)
	score, acc = model.evaluate(x_test, y_test, verbose=0)
	print('Test accuracy:', acc)
	return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
	best_run, best_model = optim.minimize(model=create_model,
	                                      data=data,
	                                      algo=tpe.suggest,
	                                      max_evals=5,
	                                      trials=Trials())
	X_train, Y_train, X_test, Y_test = data()
	print("Evalutation of best performing model:")
	print(best_model.evaluate(X_test, Y_test))
	print("Best performing model chosen hyper-parameters:")
	print(best_run)