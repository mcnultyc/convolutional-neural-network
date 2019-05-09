from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.preprocessing import image

import numpy as np
import matplotlib.pyplot as plt

classifier = Sequential()

# build the first convolutional layer
classifier.add(Conv2D(filters=96,
                      input_shape=(224, 224, 3),
                      kernel_size=(11, 11),
                      strides=(4, 4),
                      padding='valid',
                      activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
classifier.add(BatchNormalization())

# second convolutional layer
classifier.add(Conv2D(filters=256,
                      kernel_size=(11, 11),
                      strides=(1, 1),
                      padding='valid',
                      activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
classifier.add(BatchNormalization())

# thid convolutional layer
classifier.add(Conv2D(filters=384,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='valid',
                      activation='relu'))
classifier.add(BatchNormalization())

# forth convolutional layer
classifier.add(Conv2D(filters=384,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='valid',
                      activation='relu'))
classifier.add(BatchNormalization())

# fifth convolutional layer
classifier.add(Conv2D(filters=256,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='valid',
                      activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
classifier.add(BatchNormalization())

# pass it to normal neural netwok now
classifier.add(Flatten())
classifier.add(Dense(4096, input_shape=(224*224*3,),  activation='relu'))
classifier.add(Dropout(0.4))
classifier.add(BatchNormalization())

# second dense layer
classifier.add(Dense(4096, activation='relu'))
classifier.add(Dropout(0.4))
classifier.add(BatchNormalization())

# third dense layer
classifier.add(Dense(1000, activation='relu'))
classifier.add(Dropout(0.4))
classifier.add(BatchNormalization())

# output layer
classifier.add(Dense(units=5, activation='softmax'))

classifier.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# preprocess the image and run the model
train_datagen = ImageDataGenerator(
    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'data/train', target_size=(224, 224), batch_size=32, class_mode='categorical')

test_set = test_datagen.flow_from_directory(
    'data/test', target_size=(224, 224), batch_size=32, class_mode='categorical')

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
