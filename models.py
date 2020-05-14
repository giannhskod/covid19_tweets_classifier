from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers

def MLP(train_x_sub):
    
    n                       = train_x_sub.shape[1]
    out_size                = 15
    dense_size              = 128
    nb_epoch                = 20
    batch_size              = 64
    print('Build model...')
        
    model = Sequential()
    model.add(Dense(dense_size, input_dim=n, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(dense_size, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(15, activation='softmax'))
    print(model.summary())
    adam = optimizers.Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['categorical_accuracy'])

    return model