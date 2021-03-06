from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from keras.preprocessing import image

classifier = Sequential()  # intialize our neural network as a seuqnetial network

# First we are adding a sequential model a convolution layer
# The frist parameter in COnv2D is the number of filters(32)
# the second parameter is the shape of each filter (3x3)
# The third parameter is the type of image that is going to be taken in. 64, 64 size, and 3 means RGB
# The fourth parameter is activation function
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# perform pooling operation on the resultant feature maps we get after the convolution operation is done on an image.
# the key is that we are trying to reduce the total number of nodes for the upcoming layers
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Taking the 2-D array i.e pooled image pixels and converting them to a one dimensional single vector.
classifier.add(Flatten())

# here we add a full connected layer  to this layer we are going to connect the set of nodes we got after the flattening step, these nodes will act as an input layer to these fully-connected layers
classifier.add(Dense(units=128, activation='relu'))

# initialize our output layer. For now it is a single node, due to binary classification
classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Perform image prepocessing

train_datagen = ImageDataGenerator(
    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'training_set', target_size=(64, 64), batch_size=32, class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'test_set', target_size=(64, 64), batch_size=32, class_mode='binary')

# fit the model according to the data
classifier.fit_generator(training_set, steps_per_epoch=8000,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000)

# Making new predictions
test_image = image.load_img("dog.jpg", target_size=(64, 64))
test_image - image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)

if result[0][0] == 1:
    prediction = 'dog'
    print("dog")
else:
    prediction = 'cat'
    print("dog")
