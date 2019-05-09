from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.preprocessing import image


# create sequential model with keras
model = Sequential()
# input layer
#model.add(Conv2D(32, (3, 3), input_shape=(192, 192, 3), activation='relu'))
model.add(Flatten(input_shape=(64, 64, 3)))
model.add(Dense(64, activation="relu", input_shape=(4096, )))
# hidden layers
model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(64, activation="relu"))
# output layer
model.add(Dense(5, activation="softmax"))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
print("Model compiled")

train_datagen = ImageDataGenerator(
    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(
    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
    'data/train', target_size=(64, 64), batch_size=16, class_mode='categorical')

test_set = test_datagen.flow_from_directory(
    'data/test', target_size=(64, 64), batch_size=16, class_mode='categorical')

cb = EarlyStopping(monitor='val_acc',
                   min_delta=0,
                   patience=0,
                   verbose=0, mode='auto')

NUM_EPOCHS = 10
# fit the model according to the data
H = model.fit_generator(training_set, steps_per_epoch=3700,
                        epochs=NUM_EPOCHS,
                        validation_data=test_set,
                        validation_steps=939)
N = NUM_EPOCHS

print(H.history)
