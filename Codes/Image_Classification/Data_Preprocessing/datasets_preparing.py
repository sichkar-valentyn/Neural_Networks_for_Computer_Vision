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
# Loading all batches and concatenating them all together
# Plotting first 100 examples of images from 10 different classes
# Pre-processing loaded CIFAR-10 datasets for further using in classifier
# Saving datasets into file


"""Importing library for object serialization
which we'll use for saving and loading serialized models"""
import pickle

# Importing other standard libraries
import numpy as np
import os
import matplotlib.pyplot as plt


# Creating function for loading single batch of CIFAR-10 dataset
def single_batch_cifar10(file):
    # Opening file for reading in binary mode
    with open(file, 'rb') as f:
        d = pickle.load(f, encoding='latin1')  # dictionary type, we use 'latin1' for python3
        x = d['data']  # numpy.ndarray type, (10000, 3072)
        y = d['labels']  # list type
        """Initially every batch's dictionary with key 'data' has shape (10000, 3072)
        Where, 10000 - number of image samples
        3072 - three channels of image (red + green + blue)
        Every row contains an image 32x32 pixels with its three channels"""
        # Here we reshape and transpose ndarray for further use
        # At the same time method 'astype()' used for converting ndarray from int to float
        # It is used further in function 'pre_process_cifar10' as it is needed to subtract float from float
        # And for standard deviation as it is needed to divide float by float
        x = x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')  # (10000, 32, 32, 3)
        # Making numpy array from list of labels
        y = np.array(y)

        # Returning ready data
        return x, y


# Creating function for loading whole CIFAR-10 dataset
def whole_cifar10():
    # Defining lists for adding all batch's data all together
    x_collect = []
    y_collect = []

    # Defining lists for loading current batch
    x, y = [], []

    # Loading all 5 batches for training and appending them together
    for i in range(1, 6):
        # Preparing current filename
        filename = os.path.join('datasets/cifar-10-batches-py', 'data_batch_' + str(i))
        # Loading current batch
        x, y = single_batch_cifar10(filename)
        # Appending data from current batch to lists
        x_collect.append(x)
        y_collect.append(y)

    # Concatenating collected data as list of lists as one list
    x_train = np.concatenate(x_collect)  # (50000, 32, 32, 3)
    y_train = np.concatenate(y_collect)  # (50000,)

    # Releasing memory from non-needed anymore arrays
    del x, y

    # Loading data for testing
    filename = os.path.join('datasets/cifar-10-batches-py', 'test_batch')
    x_test, y_test = single_batch_cifar10(filename)

    # Returning whole CIFAR-10 data for training and testing
    return x_train, y_train, x_test, y_test


# Creating function for pre-processing CIFAR-10 datasets for further use in classifier
def pre_process_cifar10():
    # Loading whole CIFAR-10 datasets
    x_train, y_train, x_test, y_test = whole_cifar10()

    # Preparing data for training, validation and testing
    # Data for testing is taken with first 1000 examples from testing dataset
    x_test = x_test[range(1000)]  # (1000, 32, 32, 3)
    y_test = y_test[range(1000)]  # (1000,)
    # Data for validation is taken with 1000 examples from training dataset in range from 49000 to 50000
    x_validation = x_train[range(49000, 50000)]  # (1000, 32, 32, 3)
    y_validation = y_train[range(49000, 50000)]  # (1000,)
    # Data for training is taken with first 49000 examples from training dataset
    x_train = x_train[range(49000)]  # (49000, 32, 32, 3)
    y_train = y_train[range(49000)]  # (49000,)

    # Normalizing data by subtracting mean image.
    # Subtracting the dataset by mean image serves to center the data.
    # It helps for each feature to have a similar range and gradients don't go out of control.
    # Calculating mean image from training dataset along the rows by specifying 'axis=0'
    mean_image = np.mean(x_train, axis=0)  # numpy.ndarray (32, 32, 3)

    # Saving calculated 'mean_image' into 'pickle' file
    # We will use it when preprocess input data for classifying
    # We will need to subtract input image for classifying
    # As we're doing now for training, validation and testing data
    dictionary = {'mean_image': mean_image}
    with open('mean_and_std.pickle', 'wb') as f:
        pickle.dump(dictionary, f)

    # Subtracting calculated mean image from pre-processed datasets
    x_train -= mean_image
    x_validation -= mean_image
    x_test -= mean_image

    # Transposing every dataset to make channels come first
    # With method copy()
    x_train = x_train.transpose(0, 3, 1, 2).copy()  # (49000, 3, 32, 32)
    x_test = x_test.transpose(0, 3, 1, 2).copy()  # (1000, 3, 32, 32)
    x_validation = x_validation.transpose(0, 3, 1, 2).copy()  # (1000, 3, 32, 32)

    # Returning result as dictionary
    d = {'x_train': x_train, 'y_train': y_train,
         'x_validation': x_validation, 'y_validation': y_validation,
         'x_test': x_test, 'y_test': y_test}

    # Returning dictionary
    return d


# Creating function for plotting examples from CIFAR-10 dataset
def plot_cifar10_examples(x_train, y_train):
    # Preparing labels for each class
    # CIFAR-10 has 10 classes from 0 to 9
    labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Taking first ten different (unique) training images from training set
    # Going through labels and putting their indexes into list
    # Starting from '0' index
    i = 0
    # Defining variable for counting total amount of examples
    m = 0
    # Defining dictionary for storing unique label numbers and their indexes
    # As key there is unique label
    # As value there is a list with indexes of this label
    d = {}
    while True:
        # Checking if label is already in dictionary
        if y_train[i] not in d:
            d[y_train[i]] = [i]
            m += 1
        # Else if label is already in dictionary adding index to the list
        elif len(d[y_train[i]]) < 10:
            d[y_train[i]] += [i]
            m += 1
        # Checking if there is already ten labels for all labels
        if m == 100:
            break
        # Increasing 'i'
        i += 1

    # Preparing figures for plotting
    figure_1, ax = plt.subplots(nrows=10, ncols=10)
    # 'ax 'is as (2, 5) np array and we can call each time ax[0, 0]

    # Plotting first ten labels of training examples
    for i in range(10):
        ax[0, i].imshow(x_train[d[i][0]])
        ax[0, i].set_axis_off()
        ax[0, i].set_title(labels[i])

    # Plotting 90 rest of training examples
    for i in range(1, 10):
        for j in range(10):
            ax[i, j].imshow(x_train[d[j][i]])
            ax[i, j].set_axis_off()

    # Giving the name to the window with figure
    figure_1.canvas.set_window_title('CIFAR-10 examples')
    # Showing the plots
    plt.show()


# Showing pre-processed data from dictionary
data = pre_process_cifar10()
for i, j in data.items():
    print(i + ':', j.shape)

# y_train: (49000,)
# x_train: (49000, 3, 32, 32)
# x_validation: (1000, 3, 32, 32)
# y_validation: (1000,)
# x_test: (1000, 3, 32, 32)
# y_test: (1000,)

# Saving loaded and preprocessed data into 'pickle' file
with open('data.pickle', 'wb') as f:
    pickle.dump(data, f)

# Checking if preprocessed data is the same with saved data into file
Opening file for reading in binary mode
with open('data.pickle', 'rb') as f:
    d = pickle.load(f, encoding='latin1')  # dictionary type, we use 'latin1' for python3

# Showing loaded data from file
for i, j in d.items():
    print(i + ':', j.shape)

# y_train: (49000,)
# x_train: (49000, 3, 32, 32)
# x_validation: (1000, 3, 32, 32)
# y_validation: (1000,)
# x_test: (1000, 3, 32, 32)
# y_test: (1000,)

# Comparing if they are the same
print(np.array_equal(data['x_train'], d['x_train']))  # True
print(np.array_equal(data['y_train'], d['y_train']))  # True
print(np.array_equal(data['x_test'], d['x_test']))  # True
print(np.array_equal(data['y_test'], d['y_test']))  # True
print(np.array_equal(data['x_validation'], d['x_validation']))  # True
print(np.array_equal(data['y_validation'], d['y_validation']))  # True

# Plotting 100 examples of training images from 10 classes
# We can't use here data after preprocessing
# Consequently, we need to use 'whole_cifar10' function
x, y, x1, y1 = whole_cifar10()
plot_cifar10_examples(x.astype('int'), y.astype('int'))

# Checking preprocessed and saved data for mean image and standard deviation
# Opening file for reading in binary mode
with open('mean_and_std.pickle', 'rb') as f:
    mean_and_std = pickle.load(f, encoding='latin1')  # dictionary type, we use 'latin1' for python3

# Showing loaded data from file
for i, j in mean_and_std.items():
    print(i + ':', j.shape)

# mean_image: (32, 32, 3)
# std: (32, 32, 3)
