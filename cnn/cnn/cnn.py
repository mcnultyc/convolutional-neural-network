import os
from os import sys
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


if __name__ == "__main__":

    path = os.path.dirname(os.path.abspath(__file__))
    path = path.replace(os.sep, '\\')

    preview_path = path + '\\preview'
    os.makedirs(preview_path, exist_ok=True) 

    train_data_generator = ImageDataGenerator(rescale=1.0/255,
                                        rotation_range=40, 
                                        shear_range=0.2, 
                                        zoom_range=0.2,
                                        horizontal_flip=True)
    test_data_generator = ImageDataGenerator(rescale=1.0/255)

    train_generator = train_data_generator.flow_from_directory(
        path + '\\data\\train',
        seed=100,
        target_size=(150,150),
        batch_size=32,
        shuffle=True,
        class_mode='binary',
        save_to_dir= preview_path,
        save_prefix='image',
        save_format='jpeg')

    test_generator = test_data_generator.flow_from_directory(
        path + '\\data\\test',
        seed=100,
        target_size=(150,150),
        batch_size=32,
        shuffle=True,
        class_mode='binary')
    
    for i in range(10):
        train_generator.next()