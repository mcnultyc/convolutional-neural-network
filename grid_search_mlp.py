import numpy as np
import h5py
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.layers import Flatten
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier

def create_model():
    # create sequential model with keras
    model = Sequential()
    # input layer
    model.add(Dense(64, activation = "relu", input_shape=(4096, )))
    #model.add(Flatten())
    # hidden layers
    model.add(Dropout(0.65))
    model.add(Dense(64, activation = "relu"))
    model.add(Dropout(0.65))
    model.add(Dense(64, activation = "relu"))
    # output layer
    model.add(Dense(5, activation = "sigmoid"))
    # compiling the model
    model.compile(
     optimizer = "adam",
     loss = "categorical_crossentropy",
     metrics = ["accuracy"]
    )
    return model


def test_model(X_train, y_train, X_test, y_test, epochs, batch_size, model):
    # train the model
    results = model.fit(
        X_train, y_train,
        epochs= epochs,
        batch_size = batch_size,
        validation_data=(X_test, y_test)
    )
    # plot the training and testing accuracy
    plt.plot(np.arange(0, epochs), results.history["acc"], label="train_acc", linewidth=2.0, color="orange")
    plt.plot(np.arange(0, epochs), results.history["val_acc"], label="val_acc", linewidth=2.0, color="green")
    plt.title("MLP Accuracy Without Standardization")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig("model_accuracy.png")
    plt.clf()


if __name__ == "__main__":
    # load the dataset
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
    # standardize images
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)
    # one-hot encode the targets
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    # create model
    model = KerasClassifier(build_fn=create_model, batch_size=500, epochs=25)
    # create subsets of parameters
    epochs = [2, 25, 50, 200]
    batch_sizes = [100, 300, 500, 700]
    # set parameters for grid search
    param_grid = dict(epochs=epochs, batch_size=batch_sizes)
    # perform grid search on hyper-parameters
    grid = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, cv=10)
    grid_result = grid.fit(X_train, y_train)
    # print score of best performing model
    print("best score: {}, using {}".format(grid_result.best_score_,grid_result.best_params_))
    # get calculated means from grid results
    means = grid_result.cv_results_["mean_test_score"]
    # get the parameters searched
    params = grid_result.cv_results_["params"]
    graphs = dict((epoch, []) for epoch in epochs)
    # store the mean values for the batch sizes of each epoch
    for mean, std, param in zip(means, stds, params):
        graphs[param["epochs"]].append(mean)
    # plot the batch sizes against the accuracies
    plt.plot(batch_sizes, graphs[epochs[0]], marker="o", color="blue", linestyle="dashed")
    plt.plot(batch_sizes, graphs[epochs[1]], marker="^", color="orange", linestyle="dashed")
    plt.plot(batch_sizes, graphs[epochs[2]], marker="*", color="magenta", linestyle="dashed")
    plt.plot(batch_sizes, graphs[epochs[3]], marker="s", color="green", linestyle="dashed")
    plt.title('MLP Mean Accuracy of Cross-Validation')
    plt.ylabel('Mean Accuracy')
    plt.xlabel('Batch Size')
    plt.legend([str(epoch) + " epoch(s)" for epoch in epochs], loc='upper left')
    plt.savefig("model_accuracy-epochs.png")
    plt.clf()
