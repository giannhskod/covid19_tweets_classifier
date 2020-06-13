from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Flatten
from keras import optimizers
from keras import metrics
from metrics import *

def MLP(train_x_sub):
    max_words = 1000
    max_len = 150
    model = Sequential()
    model.add(Embedding(max_words, 150, input_length = max_len))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(15, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(momentum=0.9,clipnorm=1.3),
              metrics=['categorical_accuracy'])

    return model