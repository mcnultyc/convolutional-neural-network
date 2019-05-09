from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image

classifier = Sequential()  # intialize our neural network as a seuqnetial network

# First we are adding a sequential model a convolution layer
# The frist parameter in COnv2D is the number of filters(32)
# the second parameter is the shape of each filter (3x3)
# The third parameter is the type of image that is going to be taken in. 64, 64 size, and 3 means RGB
# The fourth parameter is activation function
classifier.add(Conv2D(32, (3, 3), input_shape=(
    192, 192, 3), activation='relu'))

# perform pooling operation on the resultant feature maps we get after the convolution operation is done on an image.
# the key is that we are trying to reduce the total number of nodes for the upcoming layers
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(64, (3, 3),  activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))


# Taking the 2-D array i.e pooled image pixels and converting them to a one dimensional single vector.
classifier.add(Flatten())

# here we add a full connected layer  to this layer we are going to connect the set of nodes we got after the flattening step, these nodes will act as an input layer to these fully-connected layers
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=128, activation='relu'))


# initialize our output layer. For now it is a single node, due to binary classification
classifier.add(Dense(units=5, activation='softmax'))

classifier.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Perform image prepocessing

train_datagen = ImageDataGenerator(
    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'data/train', target_size=(192, 192), batch_size=16, class_mode='categorical')

test_set = test_datagen.flow_from_directory(
    'data/test', target_size=(192, 192), batch_size=16, class_mode='categorical')

cb = EarlyStopping(monitor='val_acc',
                   min_delta=0,
                   patience=0,
                   verbose=0, mode='auto')

NUM_EPOCHS = 10
# fit the model according to the data
H = classifier.fit_generator(training_set, steps_per_epoch=3700,
                             epochs=NUM_EPOCHS,
                             validation_data=test_set,
                             validation_steps=939,
                             callbacks=[cb])
N = NUM_EPOCHS

print(H.history)
