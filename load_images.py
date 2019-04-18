import os
import getopt, sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image
from matplotlib import pyplot


def load_data(data_path):
    # get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_path) 
                   if os.path.isdir(os.path.join(data_path, d))]
    # loop through the label directories and collect the data in
    # two lists, labels and images
    labels = []
    images = []
    label = 0
    for d in directories:
        label_dir = os.path.join(data_path, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) 
                      if f.endswith(".jpg")]
        for f in file_names:
            image = load_img(f, target_size=(150, 150))
            images.append(img_to_array(image))
            labels.append(label)
        label = label+1
    return np.array(images), np.array(labels)


if __name__ == "__main__":
    # delete before submitting
    sys.argv.append('--path')
    sys.argv.append(r'C:\Users\carlo\Documents\UIC\S19\CS412\project\data')
    try:
        # parse command line arguments
        opts, args = getopt.getopt(sys.argv[1:], "p:h", ["path=", "help"])
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        sys.exit(1)
    # set default path to data directory inside of working directory
    path = os.getcwd()
    data_path = os.path.join(path, "data")
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            exit(0)
        elif opt in ("-p", "--path"):
            data_path = arg
        else:
            assert False, "invalid option"
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")
    images, labels = load_data(train_path)

    generator = ImageDataGenerator(zca_whitening=True,
                                   horizontal_flip=True,
                                   rotation_range=20,
                                   featurewise_center=True,
                                   featurewise_std_normalization=True)
    # calculate statistics on train dataset
    generator.fit(images)
    print("mean: {}".format(generator.mean))
    print("std dev: {}".format(generator.std))
    for X_batch, y_batch in generator.flow(images, labels, batch_size=32):
	    # create a grid of 3x3 images
	    for i in range(0, 9):
	        pyplot.subplot(330 + 1 + i)
	        pyplot.imshow(X_batch[i])   
	    # show the plot
	    pyplot.show()
	    break
