import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical


if __name__ == "__main__":
    h5f = h5py.File('dog_data.h5','r')
    data = h5f['dataset_1'][:]
    np.set_printoptions(suppress=True, precision=6)
    # separate images and labels
    X = data[:,:-1]
    y = data[:,-1]
    # split training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.3, 
                                                        random_state=100)
    X_train = X_train.reshape(X_train.shape[0], 64, 64, 3)
    X_test = X_test.reshape(X_test.shape[0], 64, 64, 3)
    # convert images to grayscale
    X_train = 0.7 * X_train[:,:,:,2] + 0.72 * X_train[:,:,:,1] + 0.21 * X_train[:,:,:,0]
    X_test = 0.7 * X_test[:,:,:,2] + 0.72 * X_test[:,:,:,1] + 0.21 * X_test[:,:,:,0]
    # reshape images to single dimension
    X_train = X_train.reshape(X_train.shape[0], 64 * 64)
    X_test = X_test.reshape(X_test.shape[0], 64 * 64)
    # one-hot encode the targets
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    # create sequential model with keras
    model = Sequential()
    # input layer
    model.add(Dense(64, activation = "relu", input_shape=(4096, )))
    # hidden layers
    model.add(Dropout(0.5, noise_shape=None, seed=None))
    model.add(Dense(64, activation = "relu"))
    model.add(Dropout(0.5, noise_shape=None, seed=None))
    model.add(Dense(64, activation = "relu"))
    # output layer
    model.add(Dense(5, activation = "sigmoid"))
    model.summary()
    # compiling the model
    model.compile(
     optimizer = "adam",
     loss = "binary_crossentropy",
     metrics = ["accuracy"]
    )
    # train the model
    results = model.fit(
     X_train, y_train,
     epochs= 25,
     batch_size = 500,
    )
    # score the model
    score = model.evaluate(X_test, y_test, batch_size=128)
    # score metric names are ['loss', 'acc'], accuracy is second value
    print("accuracy: {}".format(score[1]))
