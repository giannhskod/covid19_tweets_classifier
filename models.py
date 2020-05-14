from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers

def MLP(train_x_sub):
    
    n                       = train_x_sub.shape[1]
    out_size                = 15
    dense_size              = 512
    nb_epoch                = 20
    batch_size              = 256
    print('Build model...')
        
    model = Sequential()
    model.add(Dense(dense_size, input_dim=n, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(dense_size, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(dense_size, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(dense_size, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(dense_size, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(15, activation='softmax'))
    print(model.summary())
    adam = optimizers.SGD(learning_rate=0.0005, momentum=0.0, nesterov=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    return model