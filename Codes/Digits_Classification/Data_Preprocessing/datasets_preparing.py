# File: datasets_preparing.py
# Description: Neural Networks for computer vision in autonomous vehicles and robotics
# Environment: PyCharm and Anaconda environment
#
# MIT License
# Copyright (c) 2018 Valentyn N Sichkar
# github.com/sichkar-valentyn
#
# Reference to:
# Valentyn N Sichkar. Neural Networks for computer vision in autonomous vehicles and robotics // GitHub platform. DOI: 10.5281/zenodo.1317904




# Preparing datasets for further using
# Plotting first 100 examples of images with digits from 10 different classes
# Pre-processing loaded MNIST datasets for further using in classifier
# Saving datasets into file


"""Importing library for object serialization
which we'll use for saving and loading serialized models"""
import pickle

# Importing other standard libraries
import gzip
import numpy as np
import matplotlib.pyplot as plt


# Creating function for loading MNIST images
def load_data(file, number_of_images):
    # Opening file for reading in binary mode
    with gzip.open(file) as bytestream:
        bytestream.read(16)
        """Initially testing file with images has shape (60000 * 784)
        Where, 60000 - number of image samples
        784 - one channel of image (28 x 28)
        Every image consists of 28x28 pixels with its only one channel"""
        # Reading data
        buf = bytestream.read(number_of_images * 28 * 28)
        # Placing data in numpy array and converting it into 'float32' type
        # It is used further in function 'pre_process_mnist' as it is needed to subtract float from float
        # And for standard deviation as it is needed to divide float by float
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        # Reshaping data making for every image separate matrix (28, 28)
        data = data.reshape(number_of_images, 28, 28)  # (60000, 28, 28)

        # Preparing array with shape for 1 channeled image
        # Making for every image separate matrix (28, 28, 1)
        array_of_image = np.zeros((number_of_images, 28, 28, 1))  # (60000, 28, 28, 1)

        # Assigning to array one channeled image from dataset
        # In this way we get normal 3-channeled images
        array_of_image[:, :, :, 0] = data

    # Returning array of loaded images from file
    return array_of_image


# Creating function for loading MNIST labels
def load_labels(file, number_of_labels):
    # Opening file for reading in binary mode
    with gzip.open(file) as bytestream:
        bytestream.read(8)
        """Initially testing file with labels has shape (60000)
        Where, 60000 - number of labels"""
        # Reading data
        buf = bytestream.read(number_of_labels)
        # Placing data in numpy array and converting it into 'int64' type
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)  # (60000, )

    # Returning array of loaded labels from file
    return labels


# Creating function for pre-processing MNIST datasets for further use in classifier
def pre_process_mnist(x_train, y_train, x_test, y_test):
    # Normalizing whole data by dividing /255.0
    x_train /= 255.0
    x_test /= 255.0  # Data for testing consists of 10000 examples from testing dataset

    # Preparing data for training, validation and testing
    # Data for validation is taken with 1000 examples from training dataset in range from 59000 to 60000
    batch_mask = list(range(59000, 60000))
    x_validation = x_train[batch_mask]  # (1000, 28, 28, 1)
    y_validation = y_train[batch_mask]  # (1000,)
    # Data for training is taken with first 59000 examples from training dataset
    batch_mask = list(range(59000))
    x_train = x_train[batch_mask]  # (59000, 28, 28, 1)
    y_train = y_train[batch_mask]  # (59000,)

    # Normalizing data by subtracting mean image and dividing by standard deviation
    # Subtracting the dataset by mean image serves to center the data.
    # It helps for each feature to have a similar range and gradients don't go out of control.
    # Calculating mean image from training dataset along the rows by specifying 'axis=0'
    mean_image = np.mean(x_train, axis=0)  # numpy.ndarray (28, 28, 1)

    # Calculating standard deviation from training dataset along the rows by specifying 'axis=0'
    std = np.std(x_train, axis=0)  # numpy.ndarray (28, 28, 1)
    # Taking into account that a lot of values are 0, that is why we need to replace it to 1
    # In order to avoid dividing by 0
    for j in range(28):
        for i in range(28):
            if std[i, j, 0] == 0:
                std[i, j, 0] = 1.0

    # Saving calculated 'mean_image' and 'std' into 'pickle' file
    # We will use them when preprocess input data for classifying
    # We will need to subtract and divide input image for classifying
    # As we're doing now for training, validation and testing data
    dictionary = {'mean_image': mean_image, 'std': std}
    with open('mean_and_std.pickle', 'wb') as f_mean_std:
        pickle.dump(dictionary, f_mean_std)

    # Subtracting calculated mean image from pre-processed datasets
    x_train -= mean_image
    x_validation -= mean_image
    x_test -= mean_image

    # Dividing then every dataset by standard deviation
    x_train /= std
    x_validation /= std
    x_test /= std

    # Transposing every dataset to make channels come first
    x_train = x_train.transpose(0, 3, 1, 2)  # (59000, 1, 28, 28)
    x_test = x_test.transpose(0, 3, 1, 2)  # (10000, 1, 28, 28)
    x_validation = x_validation.transpose(0, 3, 1, 2)  # (10000, 1, 28, 28)

    # Returning result as dictionary
    d_processed = {'x_train': x_train, 'y_train': y_train,
                   'x_validation': x_validation, 'y_validation': y_validation,
                   'x_test': x_test, 'y_test': y_test}

    # Returning dictionary
    return d_processed


# Creating function for plotting examples from CIFAR-10 dataset
def plot_mnist_examples(x_train, y_train):
    # Preparing labels for each class
    # MNIST has 10 classes from 0 to 9
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # Taking first ten different (unique) training images from training set
    # Going through labels and putting their indexes into list
    # Starting from '0' index
    i = 0
    # Defining variable for counting total amount of examples
    m = 0
    # Defining dictionary for storing unique label numbers and their indexes
    # As key there is unique label
    # As value there is a list with indexes of this label
    d_plot = {}
    while True:
        # Checking if label is already in dictionary
        if y_train[i] not in d_plot:
            d_plot[y_train[i]] = [i]
            m += 1
        # Else if label is already in dictionary adding index to the list
        elif len(d_plot[y_train[i]]) < 10:
            d_plot[y_train[i]] += [i]
            m += 1
        # Checking if there is already ten labels for all labels
        if m == 100:
            break
        # Increasing 'i'
        i += 1

    # Preparing figures for plotting
    figure_1, ax = plt.subplots(nrows=10, ncols=10)
    # 'ax 'is as (10, 10) np array and we can call each time ax[0, 0]

    # Plotting first ten labels of training examples
    # Here we plot only matrix of image with only one channel '[:, :, 0]'
    # Showing image in grayscale specter by 'cmap=plt.get_cmap('gray')'
    for i in range(10):
        ax[0, i].imshow(x_train[d_plot[i][0]][:, :, 0], cmap=plt.get_cmap('gray'))
        ax[0, i].set_axis_off()
        ax[0, i].set_title(labels[i])

    # Plotting 90 rest of training examples
    # Here we plot only matrix of image with only one channel '[:, :, 0]'
    # Showing image in grayscale specter by 'cmap=plt.get_cmap('gray')'
    for i in range(1, 10):
        for j in range(10):
            ax[i, j].imshow(x_train[d_plot[j][i]][:, :, 0], cmap=plt.get_cmap('gray'))
            ax[i, j].set_axis_off()

    # Giving the name to the window with figure
    figure_1.canvas.set_window_title('MNIST examples')
    # Showing the plots
    plt.show()


# Creating function for plotting 35 images with random images from MNIST dataset
# After images are created, we will assemble them and make animated .gif image
def plot_mnist_35_images(x_train, y_train):
    # Making batch from training data with random data
    # Getting total number of training images
    number_of_training_images = x_train.shape[0]
    # Getting random batch of 'batch_size' size from total number of training images
    batch_size = 1000
    batch_mask = np.random.choice(number_of_training_images, batch_size)
    # Getting training dataset according to the 'batch_mask'
    x_train = x_train[batch_mask]
    y_train = y_train[batch_mask]

    # Preparing labels for each class
    # MNIST has 10 classes from 0 to 9
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # Taking first ten different (unique) training images from training set
    # Going through labels and putting their indexes into list
    # Starting from '0' index
    i = 0
    # Defining variable for counting total amount of examples
    m = 0
    # Defining dictionary for storing unique label numbers and their indexes
    # As key there is unique label
    # As value there is a list with indexes of this label
    d_plot = {}
    while True:
        # Checking if label is already in dictionary
        if y_train[i] not in d_plot:
            d_plot[y_train[i]] = [i]
            m += 1
        # Else if label is already in dictionary adding index to the list
        # In order to create 35 images we need 44 indexes of every class
        elif len(d_plot[y_train[i]]) < 44:
            d_plot[y_train[i]] += [i]
            m += 1
        # Checking if there is already 440 indexes for all labels
        if m == 440:
            break
        # Increasing 'i'
        i += 1

    # Creating 35 images by showing first image
    # And then deleting first line and upping data
    # Preparing figures for plotting
    figure_1, ax = plt.subplots(nrows=10, ncols=10)
    # 'ax 'is as (10, 10) np array and we can call each time ax[0, 0]
    for k in range(35):
        # Plotting first ten labels of training examples
        # Here we plot only matrix of image with only one channel '[:, :, 0]'
        # Showing image in grayscale specter by 'cmap=plt.get_cmap('gray')'
        for i in range(10):
            ax[0, i].imshow(x_train[d_plot[i][k]][:, :, 0], cmap=plt.get_cmap('gray'))
            ax[0, i].set_axis_off()
            ax[0, i].set_title(labels[i])

        # Plotting 90 rest of training examples
        # Here we plot only matrix of image with only one channel '[:, :, 0]'
        # Showing image in grayscale specter by 'cmap=plt.get_cmap('gray')'
        for i in range(1, 10):
            for j in range(10):
                ax[i, j].imshow(x_train[d_plot[j][i + k]][:, :, 0], cmap=plt.get_cmap('gray'))
                ax[i, j].set_axis_off()

        # Creating unique name for the plot
        name_of_plot = str(k) + '_feeding_with_mnist.png'
        # Setting path for saving image
        path_for_saving_plot = 'feeding_images/' + name_of_plot
        # Saving plot
        figure_1.savefig(path_for_saving_plot)
        plt.close()


# Plotting 100 examples of training images from 10 classes
# We can't use here data after preprocessing
x = load_data('datasets/train-images-idx3-ubyte.gz', 1000)  # (1000, 28, 28, 1)
y = load_labels('datasets/train-labels-idx1-ubyte.gz', 1000)  # (1000,)
# Also, making arrays as type of 'int' in order to show correctly on the plot
plot_mnist_examples(x.astype('int'), y.astype('int'))


# Loading whole data for preprocessing
x_train = load_data('datasets/train-images-idx3-ubyte.gz', 60000)
y_train = load_labels('datasets/train-labels-idx1-ubyte.gz', 60000)
x_test = load_data('datasets/t10k-images-idx3-ubyte.gz', 1000)
y_test = load_labels('datasets/t10k-labels-idx1-ubyte.gz', 1000)
# Showing pre-processed data from dictionary
data = pre_process_mnist(x_train, y_train, x_test, y_test)
for i, j in data.items():
    print(i + ':', j.shape)

# x_train: (59000, 1, 28, 28)
# y_train: (59000,)
# x_validation: (1000, 1, 28, 28)
# y_validation: (1000,)
# x_test: (1000, 1, 28, 28)
# y_test: (1000,)

# Saving loaded and preprocessed data into 'pickle' file
with open('data0.pickle', 'wb') as f:
    pickle.dump(data, f)

