import numpy
from keras.datasets import imdb
import embedding_utils
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, CuDNNGRU, Dropout, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import keras.optimizers

# load the dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data()
X = numpy.concatenate((X_train, X_test), axis=0)
y = numpy.concatenate((y_train, y_test), axis=0)
vocab_size = len(imdb.get_word_index())
dim_size = embedding_utils.get_dim()
embedding_matrix = embedding_utils.load_embedding()
max_len = 1500 #completely arbitrary
#the outlier of the data contains around 2.7k words, I'm not gonna use it.

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

model = Sequential()
model.add(Embedding(vocab_size, dim_size, input_length=max_len, weights=[embedding_matrix], trainable=False))

model.add(CuDNNGRU(32, return_sequences=True))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(GlobalMaxPooling1D())

model.add(Dense(32))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dense(2), activation='sigmoid')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))