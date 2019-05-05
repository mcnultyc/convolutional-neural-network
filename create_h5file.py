import os
import numpy as np
import sys
import time
import h5py
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from matplotlib import pyplot

def load_data(data_path, target_size):
    # create paths for training and testing directories
    outer_directories = [os.path.join(data_path, d) for d in os.listdir(data_path) 
                       if os.path.isdir(os.path.join(data_path, d))]
    data = []
    for path in outer_directories:
        print(path)
        # get all subdirectories of the data directory
        # each represents a label.
        directories = [d for d in os.listdir(path) 
                       if os.path.isdir(os.path.join(path, d))]
        # order of class directories have to be the same
        label = 0
        for d in directories:
            label_dir = os.path.join(path, d)
            # get list of jpg files in directory
            file_names = [os.path.join(label_dir, f) 
                          for f in os.listdir(label_dir) 
                          if f.endswith(".jpg")]
            # load jpg images from directory
            for f in file_names:
                # load image
                image = load_img(f, target_size=target_size)
                # convert image to array and flatten
                image = img_to_array(image).flatten()
                # add label to image
                data.append(np.append(image, label))
            label = label+1
            time.sleep(30)
            # print directory and number of image files
            print("     directory: {} ({})".format(label_dir, len(file_names)))
    return np.array(data)

if __name__ == "__main__":
    target_size = (64, 64)
    data_path = os.path.join(os.getcwd(), "data")
    # get the data from the images
    orig_data = load_data(data_path, target_size)
    # remove the label from the first image
    orig_image = orig_data[0, :-1]
    # store the data in data.h5
    h5f = h5py.File('dog_data.h5', 'w')
    h5f.create_dataset('dataset_1', data=orig_data)
    h5f.close()
    # load the images from data.h5
    h5f = h5py.File('dog_data.h5','r')
    saved_data = h5f['dataset_1'][:]
    # remove the label from the first stored image
    saved_image = saved_data[0, :-1]
    h5f.close()
    # verify that the data matches
    print("data integrity: {}".format(np.allclose(orig_data, saved_data)))
    # reshape the original image
    orig_image = orig_image.reshape((target_size[0], target_size[1], 3))
    # reshape the saved image
    saved_image = saved_image.reshape((target_size[0], target_size[1], 3))
    # show the original image
    pyplot.imshow(array_to_img(orig_image))
    pyplot.show()
    # show the saved image
    pyplot.imshow(array_to_img(saved_image))
    pyplot.show()
