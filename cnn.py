import numpy as np
import embedding_utils
import data_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, Dropout, Activation, Conv1D, Input, Concatenate, Flatten

x_train, x_test, y_train, y_test = data_utils.load_data()
word_index = data_utils.get_wi()
vocab_size = len(word_index)
embedding_matrix = embedding_utils.load_embedding()
dim_size = embedding_utils.get_dim()
max_len = data_utils.get_max_len()

input_shape = Input(shape=(max_len, ), dtype='int32')
nb_filter = 5
nb_kernel = [3, 4, 5]
result = []

embed = Embedding(vocab_size, dim_size, input_length=max_len, weights=[embedding_matrix], trainable=False)(input_shape)
for i in nb_kernel:
	conv = Conv1D(filters=nb_filter, kernel_size=i, activation='relu')(embed)
	pool = GlobalMaxPooling1D()(conv)
	result.append(pool)

conc = Concatenate()(result)
# print(conc.shape)
# conc = Flatten()(conc)
pred = Dense(1, activation='sigmoid')(conc)

model = Model(inputs=input_shape, outputs=pred)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=200, batch_size=32, verbose=2)
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))