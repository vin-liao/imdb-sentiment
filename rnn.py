import numpy as np
import embedding_utils
import data_utils
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, CuDNNGRU, Dropout, BatchNormalization, Activation

x_train, x_test, y_train, y_test = data_utils.load_data()
word_index = data_utils.get_wi()
vocab_size = len(word_index)
embedding_matrix = embedding_utils.load_embedding()
dim_size = embedding_utils.get_dim()
max_len = data_utils.get_max_len()

model = Sequential()
model.add(Embedding(vocab_size, dim_size, input_length=max_len, weights=[embedding_matrix], trainable=False))

model.add(CuDNNGRU(20, return_sequences=True, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(GlobalMaxPooling1D())

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=32, verbose=2)
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))