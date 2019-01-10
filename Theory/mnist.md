# MNIST Digits Classification with `numpy` only
Example on Digits Classification with the help of MNIST dataset of handwritten digits and Convolutional Neural Network.
<br/>[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1317904.svg)](https://doi.org/10.5281/zenodo.1317904)

## Test online [here](https://valentynsichkar.name/mnist.html)

## Content
Theory and experimental results (on this page)

* [MNIST Digits Classification with `numpy` only](#mnist-digits-classification-with-numpy-library-only)
  * [Loading MNIST dataset](#loading-mnist-dataset)
  * [Plotting examples of digits from MNIST dataset](#plotting-examples-of-digits-from-mnist-dataset)
  * [Preprocessing loaded MNIST dataset](#preprocessing-loaded-mnist-dataset)
  * [Saving and Loading serialized models](#saving-and-loading-serialized-models)
  * [Functions for dealing with CNN layers](#functions-for-dealing-with-cnn-layers)
    * [Naive Forward Pass for Convolutional layer](#naive-forward-pass-for-convolutional-layer)
    * [Naive Backward Pass for Convolutional layer](#naive-backward-pass-for-convolutional-layer)
    * [Naive Forward Pass for Max Pooling layer](#naive-forward-pass-for-max-pooling-layer)
    * [Naive Backward Pass for Max Pooling layer](#naive-backward-pass-for-max-pooling-layer)
    * [Forward Pass for Affine layer](#forward-pass-for-affine-layer)
    * [Backward Pass for Affine layer](#backward-pass-for-affine-layer)
    * [Forward Pass for ReLU layer](#forward-pass-for-relu-layer)
    * [Backward Pass for ReLU layer](#backward-pass-for-relu-layer)
    * [Softmax Classification loss](#softmax-classification-loss)
  * [Creating Classifier - model of CNN](#creating-classifier-model-of-cnn)
    * [Initializing new Network](#initializing-new-network)
    * [Evaluating loss for training ConvNet1](#evaluating-loss-for-training-convnet1)
    * [Calculating scores for predicting ConvNet1](#calculating-scores-for-predicting-convnet1)
  * [Functions for Optimization](#optimization-functions)
    * [Vanilla SGD](#vanilla-sgd)
    * [Momentum SGD](#momentum-sgd)
    * [RMS Propagation](#rms-propagation)
    * [Adam](#adam)
  * [Creating Solver Class](#creating-solver-class)
    * [_Reset](#reset)
    * [_Step](#step)
    * [Checking Accuracy](#accuracy)
    * [Train](#train)
  * [Overfitting Small Data](#overfitting-small-data)
  * [Training Results](#training-results)
  * [Full Codes](#full-codes)
 
<br/>

### <a id="mnist-digits-classification-with-numpy-library-only">MNIST Digits Classification with `numpy` only</a>
In this example we'll test CNN for Digits Classification with the help of MNIST dataset.
<br/>Following standard and most common parameters can be used and tested:

| Parameter | Description |
| --- | --- |
| Weights Initialization | HE Normal |
| Weights Update Policy | Vanilla SGD, Momentum SGD, RMSProp, Adam |
| Activation Functions | ReLU, Sigmoid |
| Regularization | L2, Dropout |
| Pooling | Max, Average |
| Loss Functions | Softmax, SVM |

<br/>Contractions:
* **Vanilla SGD** - Vanilla Stochastic Gradient Descent
* **Momentum SGD** - Stochastic Gradient Descent with Momentum
* **RMSProp** - Root Mean Square Propagation
* **Adam** - Adaptive Moment Estimation
* **SVM** - Support Vector Machine

<br/>**For current example** following architecture will be used:
<br/>`Input` --> `Conv` --> `ReLU` --> `Pool` --> `Affine` --> `ReLU` --> `Affine` --> `Softmax`

![Model_1_Architecture.png](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Model_1_Architecture_MNIST.png)

<br/>**For current example** following parameters will be used:

| Parameter | Description |
| --- | --- |
| Weights Initialization | `HE Normal` |
| Weights Update Policy | `Vanilla SGD` |
| Activation Functions | `ReLU` |
| Regularization | `L2` |
| Pooling | `Max` |
| Loss Functions | `Softmax` |

<br/>**File structure** with folders and functions can be seen on the figure below:

![Image_Classification_File_Structure.png](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Image_Classification_Files_Structure.png)

<br/>Also, **file structure** can be seen below:
* MNIST Digits Classification with `numpy` only:
  * `Data_Preprocessing`
    * `datasets`
    * [datasets_preparing.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Data_Preprocessing/datasets_preparing.py)
    * [mean_and_std.pickle](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Data_Preprocessing/mean_and_std.pickle)    
  * `Helper_Functions`
    * [layers.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Helper_Functions/layers.py)
    * [optimize_rules.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Helper_Functions/optimize_rules.py)
  * `Classifiers`
    * [ConvNet1.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Classifiers/ConvNet1.py) 
  * `Serialized_Models`
    * model1.pickle
  * [Solver.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Solver.py)
  
<br/>

### <a id="loading-mnist-dataset">Loading MNIST dataset</a>
After downloading files from official resource, there has to be following files:
* train-images-idx3-ubyte.gz
* train-labels-idx1-ubyte.gz
* t10k-images-idx3-ubyte.gz
* t10k-labels-idx1-ubyte.gz

Writing code in Python.
<br/>Importing needed libraries.
<br/>Consider following part of the code:
<br/>(related file: [datasets_preparing.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Data_Preprocessing/datasets_preparing.py))

```py
"""Importing library for object serialization
which we'll use for saving and loading serialized models"""
import pickle

# Importing other standard libraries
import gzip
import numpy as np
import matplotlib.pyplot as plt
```

Creating function for loading MNIST images.
<br/>Consider following part of the code:
<br/>(related file: [datasets_preparing.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Data_Preprocessing/datasets_preparing.py))

```py
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
```

Creating function for loading MNIST labels.
<br/>Consider following part of the code:
<br/>(related file: [datasets_preparing.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Data_Preprocessing/datasets_preparing.py))

```py
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
```

<br/>

### <a id="plotting-examples-of-digits-from-mnist-dataset">Plotting examples of digits from MNIST dataset</a>
After dataset was load, it is possible to show examples of training images.
<br/>Creating function for showing first 100 unique example of images from MNIST dataset.
<br/>Consider following part of the code:
<br/>(related file: [datasets_preparing.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Data_Preprocessing/datasets_preparing.py))

```py
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
```

For plotting images consider following part of the code:
<br/>(related file: [datasets_preparing.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Data_Preprocessing/datasets_preparing.py))

```py
# Plotting 100 examples of training images from 10 classes
# We can't use here data after preprocessing
x = load_data('datasets/train-images-idx3-ubyte.gz', 1000)  # (1000, 28, 28, 1)
y = load_labels('datasets/train-labels-idx1-ubyte.gz', 1000)  # (1000,)
# Also, making arrays as type of 'int' in order to show correctly on the plot
plot_mnist_examples(x.astype('int'), y.astype('int'))
```

Result can be seen on the image below.

![MNIST_examples](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/MNIST_examples.png)

<br/>

### <a id="preprocessing-loaded-mnist-dataset">Preprocessing loaded MNIST dataset</a>
Next, creating function for preprocessing MNIST dataset for further use in classifier.
* Normalizing data by `dividing / 255.0` (!) - up to researcher
* Normalizing data by `subtracting mean image` and `dividing by standard deviation` (!) - up to researcher
* Transposing every dataset to make channels come first
* Returning result as dictionary

Consider following part of the code:
<br/>(related file: [datasets_preparing.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Data_Preprocessing/datasets_preparing.py))

```py
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
```

After running created function, it is possible to see loaded, prepared and preprocessed CIFAR-10 datasets.
<br/>Consider following part of the code:
<br/>(related file: [datasets_preparing.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Data_Preprocessing/datasets_preparing.py))

```py
# Loading whole data for preprocessing
x_train = load_data('datasets/train-images-idx3-ubyte.gz', 60000)
y_train = load_labels('datasets/train-labels-idx1-ubyte.gz', 60000)
x_test = load_data('datasets/t10k-images-idx3-ubyte.gz', 1000)
y_test = load_labels('datasets/t10k-labels-idx1-ubyte.gz', 1000)
# Showing pre-processed data from dictionary
data = pre_process_mnist(x_train, y_train, x_test, y_test)
for i, j in data.items():
    print(i + ':', j.shape)
```

As a result there will be following:
* `x_train: (59000, 1, 28, 28)`
* `y_train: (59000,)`
* `x_validation: (1000, 1, 28, 28)`
* `y_validation: (1000,)`
* `x_test: (1000, 1, 28, 28)`
* `y_test: (1000,)`


<br/>

### <a id="saving-and-loading-serialized-models">Saving and Loading serialized models</a>
Saving loaded and preprocessed data into 'pickle' file.
<br/>Consider following part of the code:
<br/>(related file: [datasets_preparing.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Data_Preprocessing/datasets_preparing.py))

```py
with open('data.pickle', 'wb') as f:
    pickle.dump(data, f)
```
<br/>

### <a id="functions-for-dealing-with-cnn-layers">Functions for dealing with CNN layers</a>
Creating functions for CNN layers:
* Naive Forward Pass for Convolutional layer
* Naive Backward Pass for Convolutional layer
* Naive Forward Pass for Max Pooling layer
* Naive Backward Pass for Max Pooling layer
* Forward Pass for Affine layer
* Backward Pass for Affine layer
* Forward Pass for ReLU layer
* Backward Pass for ReLU layer
* Softmax Classification loss

#### <a id="naive-forward-pass-for-convolutional-layer">Naive Forward Pass for Convolutional layer</a>
Defining function for naive forward pass for convolutional layer.
```py
"""
Input consists of following:
    x of shape (N, C, H, W) - N data, each with C channels, height H and width W.
    w of shape (F, C, HH, WW) - We convolve each input with F different filters,
        where each filter spans all C channels; each filter has height HH and width WW.
    'cnn_params' is a dictionary with following keys:
        'stride' - step for sliding,
        'pad' - zero-pad frame around input.
Function returns a tuple of (out, cash):
    feature_maps - output data of feature maps of shape (N, F, H', W') where:
        H' = 1 + (H + 2 * pad - HH) / stride
        W' = 1 + (W + 2 * pad - WW) / stride
            where,
            N here is the same as we have it as number of input images,
            F here is as number of channels of each N (that are now as feature maps).
    cache - is a tuple of (x, w, b, cnn_params), needed in backward pass.
    
"""
```

Consider following part of the code:
<br/>(related file: [layers.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Helper_Functions/layers.py))

```py
def cnn_forward_naive(x, w, b, cnn_params):
    # Preparing parameters for convolution operation
    stride = cnn_params['stride']
    pad = cnn_params['pad']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    # Cache for output
    cache = (x, w, b, cnn_params)

    # Applying to the input image volume Pad frame with zero values for all channels
    # As we have in input x N as number of inputs, C as number of channels,
    # then we don't have to pad them
    # That's why we leave first two tuples with 0 - (0, 0), (0, 0)
    # And two last tuples with pad parameter - (pad, pad), (pad, pad)
    # In this way we pad only H and W of N inputs with C channels
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

    # Defining spatial size of output image volume (feature maps) by following formulas:
    height_out = int(1 + (H + 2 * pad - HH) / stride)
    width_out = int(1 + (W + 2 * pad - WW) / stride)
    # Depth of output volume is number of filters which is F
    # And number of input images N remains the same - it is number of output image volumes now

    # Creating zero valued volume for output feature maps
    feature_maps = np.zeros((N, F, height_out, width_out))

    # Implementing convolution through N input images, each with F filters
    # Also, with respect to C channels
    # For every image
    for n in range(N):
        # For every filter
        for f in range(F):
            # Defining variable for indexing height in output feature map
            # (because our step might not be equal to 1)
            height_index = 0
            # Convolving every channel of the image with every channel of the current filter
            # Result is summed up
            # Going through all input image (2D convolution) through all channels
            for i in range(0, H, stride):
                # Defining variable for indexing width in output feature map
                # (because our step might not be equal to 1)
                width_index = 0
                for j in range(0, W, stride):
                    feature_maps[n, f, height_index, width_index] = \
                        np.sum(x_padded[n, :, i:i+HH, j:j+WW] * w[f, :, :, :]) + b[f]
                    # Increasing index for width
                    width_index += 1
                # Increasing index for height
                height_index += 1

    # Returning resulted volumes of feature maps and cash
    return feature_maps, cache
```

#### <a id="naive-backward-pass-for-convolutional-layer">Naive Backward Pass for Convolutional layer</a>
Defining function for naive backward pass for convolutional layer.
```py
"""
Defining function for naive backward pass for convolutional layer.
Input consists of following:
    derivatives_out - Upstream derivatives.
    cache - is a tuple of (x, w, b, cnn_params) as in 'cnn_forward_naive' function:
        x of shape (N, C, H, W) - N data, each with C channels, height H and width W.
        w of shape (F, C, HH, WW) - We convolve each input with F different filters,
            where each filter spans all C channels; each filter has height HH and width WW.
        'cnn_params' is a dictionary with following keys:
            'stride' - step for sliding,
            'pad' - zero-pad frame around input.
Function returns a tuple of (dx, dw, db):
    dx - gradient with respect to x,
    dw - gradient with respect to w,
    db - gradient with respect to b.
"""
```

Consider following part of the code:
<br/>(related file: [layers.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Helper_Functions/layers.py))

```py
def cnn_backward_naive(derivative_out, cache):
    # Preparing variables for input, weights, biases, cnn parameters from cache
    x, w, b, cnn_params = cache

    # Preparing variables with appropriate shapes
    N, C, H, W = x.shape  # For input
    F, _, HH, WW = w.shape  # For weights
    _, _, height_out, weight_out = derivative_out.shape  # For output feature maps

    # Preparing variables with parameters
    stride = cnn_params['stride']
    pad = cnn_params['pad']

    # Preparing gradients for output
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # It is important to remember that cash has original non-padded input x.
    # Applying to the input image volume Pad frame with zero values for all channels
    # As we have in input x N as number of inputs, C as number of channels,
    # then we don't have to pad them
    # That's why we leave first two tuples with 0 - (0, 0), (0, 0)
    # And two last tuples with pad parameter - (pad, pad), (pad, pad)
    # In this way we pad only H and W of N inputs with C channels
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
    # The same we apply padding for dx
    dx_padded = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

    # Implementing backward pass through N input images, each with F filters
    # Also, with respect to C channels
    # And calculating gradients
    # For every image
    for n in range(N):
        # For every filter
        for f in range(F):
            # Going through all input image through all channels
            for i in range(0, H, stride):
                for j in range(0, W, stride):
                    # Calculating gradients
                    dx_padded[n, :, i:i+HH, j:j+WW] += w[f, :, :, :] * derivative_out[n, f, i, j]
                    dw[f, :, :, :] += x_padded[n, :, i:i+HH, j:j+WW] * derivative_out[n, f, i, j]
                    db[f] += derivative_out[n, f, i, j]

    # Reassigning dx by slicing dx_padded
    dx = dx_padded[:, :, 1:-1, 1:-1]

    # Returning calculated gradients
    return dx, dw, db
```

#### <a id="naive-forward-pass-for-max-pooling-layer">Naive Forward Pass for Max Pooling layer</a>
Defining function for naive forward pass for Max Pooling layer.
```py
"""
Defining function for naive forward pass for Max Pooling layer.
Input consists of following:
    x as input data with shape (N, C, H, W) - N data, each with C channels, height H and width W.
    'pooling_params' is a dictionary with following keys:
        'pooling_height' - height of pooling region,
        'pooling_width' - width of pooling region,
        'stride' - step (distance) between pooling regions.
    
Function returns a tuple of (pooled_output, cache):
    pooled_output - is output resulted data of shape (N, C, H', W') where:
        H' = 1 + (H + pooling_height) / stride
        W' = 1 + (W + pooling_width) / stride
            where,
            N here is the same as we have it as number of input images,
            C here is as number of channels of each N.
    cache - is a tuple of (x, pooling_params), needed in backward pass.
"""
```

Consider following part of the code:
<br/>(related file: [layers.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Helper_Functions/layers.py))

```py
def max_pooling_forward_naive(x, pooling_params):
    # Preparing variables with appropriate shapes
    N, C, H, W = x.shape  # For input

    # Preparing variables with parameters
    pooling_height = pooling_params['pooling_height']
    pooling_width = pooling_params['pooling_width']
    stride = pooling_params['stride']

    # Cache for output
    cache = (x, pooling_params)

    # Defining spatial size of output image volume after pooling layer by following formulas:
    height_pooled_out = int(1 + (H - pooling_height) / stride)
    width_polled_out = int(1 + (W - pooling_width) / stride)
    # Depth of output volume is number of channels which is C (or number of feature maps)
    # And number of input images N remains the same - it is number of output image volumes now

    # Creating zero valued volume for output image volume after pooling layer
    pooled_output = np.zeros((N, C, height_pooled_out, width_polled_out))

    # Implementing forward naive pooling pass through N input images, each with C channels
    # And calculating output pooled image volume
    # For every image
    for n in range(N):
        # Going through all input image through all channels
        for i in range(height_pooled_out):
            for j in range(width_polled_out):
                # Preparing height and width for current pooling region
                ii = i * stride
                jj = j * stride
                # Getting current pooling region with all channels C
                current_pooling_region = x[n, :, ii:ii+pooling_height, jj:jj+pooling_width]
                # Finding maximum value for all channels C and filling output pooled image
                # Reshaping current pooling region from (3, 2, 2) - 3 channels and 2 by 2
                # To (3, 4) in order to utilize np.max function
                # Specifying 'axis=1' as parameter for choosing maximum value out of 4 numbers along 3 channels
                pooled_output[n, :, i, j] = \
                    np.max(current_pooling_region.reshape((C, pooling_height * pooling_width)), axis=1)

    # Returning output resulted data
    return pooled_output, cache
```

#### <a id="naive-backward-pass-for-max-pooling-layer">Naive Backward Pass for Max Pooling layer</a>
Defining function for naive backward pass for Max Pooling layer.
```py
"""
Defining function for naive backward pass for Max Pooling layer.
Input consists of following:
    derivatives_out - Upstream derivatives.
    cache - is a tuple of (x, pooling_params) as in 'max_pooling_forward_naive' function:
        x as input data with shape (N, C, H, W) - N data, each with C channels, height H and width W.
        'pooling_params' is a dictionary with following keys:
            'pooling_height' - height of pooling region,
            'pooling_width' - width of pooling region,
            'stride' - step (distance) between pooling regions.
    
Function returns:
    dx - gradient with respect to x.
"""
```

Consider following part of the code:
<br/>(related file: [layers.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Helper_Functions/layers.py))

```py
def max_pooling_backward_naive(derivatives_out, cache):
    # Preparing variables with appropriate shapes
    x, pooling_params = cache
    N, C, H, W = x.shape

    # Preparing variables with parameters
    pooling_height = pooling_params['pooling_height']
    pooling_width = pooling_params['pooling_width']
    stride = pooling_params['stride']

    # Defining spatial size of output image volume after pooling layer by following formulas:
    height_pooled_out = int(1 + (H - pooling_height) / stride)
    width_polled_out = int(1 + (W - pooling_width) / stride)
    # Depth of output volume is number of channels which is C (or number of feature maps)
    # And number of input images N remains the same - it is number of output image volumes now

    # Creating zero valued volume for output gradient after backward pass of pooling layer
    # The shape is the same with x.shape
    dx = np.zeros((N, C, H, W))

    # Implementing backward naive pooling pass through N input images, each with C channels
    # And calculating output pooled image volume
    # For every image
    for n in range(N):
        # For every channel
        for c in range(C):
            # Going through all pooled image by height and width
            for i in range(height_pooled_out):
                for j in range(width_polled_out):
                    # Preparing height and width for current pooling region
                    ii = i * stride
                    jj = j * stride
                    # Getting current pooling region
                    current_pooling_region = x[n, c, ii:ii+pooling_height, jj:jj+pooling_width]
                    # Finding maximum value for current pooling region
                    current_maximum = np.max(current_pooling_region)
                    # Creating array with the same shape as 'current_pooling_region'
                    # Filling with 'True' and 'False' according to the condition '==' to 'current_maximum'
                    temp = current_pooling_region == current_maximum
                    # Calculating output gradient
                    dx[n, c, ii:ii+pooling_height, jj:jj+pooling_width] += \
                        derivatives_out[n, c, i, j] * temp

                    # Backward pass for pooling layer will return gradient with respect to x
                    # Each pooling region will be filled with '0'
                    # or derivative if that value was maximum for forward pass
                    # print(x[0, 0, 0:2, 0:2])
                    # print()
                    # print(dx[0, 0, 0:2, 0:2])

                    # [[ 0.57775955 -0.03546282]
                    #  [-1.03050044 -1.23398021]]

                    # [[-0.93262122  0.        ]
                    #  [ 0.          0.        ]]

    # Returning gradient with respect to x
    return dx
```

#### <a id="forward-pass-for-affine-layer">Forward Pass for Affine layer</a>
Defining function for forward pass for Affine layer.
```py
"""
Defining function for computing forward pass for Affine layer.
Affine layer - this is Fully Connected layer.
Input consists of following:
    x - input data in form of numpy array and shape (N, d1, ..., dk),
    w - weights in form of numpy array and shape (D, M),
    b - biases in form of numpy array and shape (M,),
        where input x contains N batches and each batch x[i] has shape (d1, ..., dk).
        We will reshape each input batch x[i] into vector of dimension D = d1 * ... * dk.
        As a result, input will be in form of matrix with shape (N, D).
        It is needed for calculation product of input matrix over weights.
        As weights matrix has shape (D, M), then output resulted matrix will be with shape (N, M).
        
Function returns a tuple of:
    affine_output - output data in form of numpy array and shape (N, M),
    cache - is a tuple of (x, w, b), needed in backward pass.
"""
```

Consider following part of the code:
<br/>(related file: [layers.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Helper_Functions/layers.py))

```py
def affine_forward(x, w, b):
    # Cache for output
    cache = (x, w, b)

    # Reshaping input data with N batches into matrix with N rows
    N = x.shape[0]
    x = x.reshape(N, -1)
    # By using '-1' we say that number of column is unknown, but number of rows N is known
    # Resulted matrix will be with N rows and D columns
    # Example:
    # x = np.random.randint(0, 9, (2, 3, 3))
    # print(x.shape)  # (2, 3, 3)
    # print(x)
    #             [[[3 6 5]
    #               [6 3 2]
    #               [1 0 0]]
    #
    #              [[8 5 8]
    #               [7 5 2]
    #               [2 1 6]]]
    #
    # x = x.reshape(2, -1)
    # print(x.shape)  # (2, 9)
    # print(x)
    #             [[3 6 5 6 3 2 1 0 0]
    #              [8 5 8 7 5 2 2 1 6]]

    # Implementing Affine forward pass.
    # Calculating product of input data over weights
    affine_output = np.dot(x, w) + b

    # Returning resulted matrix with shape of (N, M)
    return affine_output, cache
```

#### <a id="backward-pass-for-affine-layer">Backward Pass for Affine layer</a>
Defining function for backward pass for Affine layer.
```py
"""
Defining function for computing backward pass for Affine layer.
Affine layer - this is Fully Connected layer.
Input consists of following:
    derivatives_out - Upstream derivatives of shape (N, M),
    cache - is a tuple of (x, w, b):
        x - input data in form of numpy array and shape (N, d1, ..., dk),
        w - weights in form of numpy array and shape (D, M),
        b - biases in form of numpy array and shape (M,).
Function returns a tuple of (dx, dw, db):
    dx - gradient with respect to x of shape (N, d1, ..., dk),
    dw - gradient with respect to w of shape (D, M),
    db - gradient with respect to b of shape (M,).
"""
```

Consider following part of the code:
<br/>(related file: [layers.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Helper_Functions/layers.py))

```py
def affine_backward(derivatives_out, cache):
    # Preparing variables for input, weights and biases from cache
    x, w, b = cache

    # Implementing backward pass for Affine layer
    # Calculating gradient with respect to x and reshaping to make shape as in x
    dx = np.dot(derivatives_out, w.T).reshape(x.shape)
    # Calculating gradient with respect to w
    # Reshaping input data with N batches into matrix with N rows and D columns
    N = x.shape[0]
    x = x.reshape(N, -1)
    dw = np.dot(x.T, derivatives_out)
    # Calculating gradient with respect to b
    db = np.dot(np.ones(dx.shape[0]), derivatives_out)
    # db = np.sum(derivatives_out, axis=0)

    # Returning calculated gradients
    return dx, dw, db
```

#### <a id="forward-pass-for-relu-layer">Forward Pass for ReLU layer</a>
Defining function for forward pass for ReLU layer.
```py
"""
Defining function for computing forward pass for ReLU layer.
ReLU layer - this is rectified linear units layer.
Input consists of following:
    x - input data of any shape.
Function returns a tuple of:
    relu_output - output data of the same shape as x,
    cache - is x, needed in backward pass.
"""
```

Consider following part of the code:
<br/>(related file: [layers.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Helper_Functions/layers.py))

```py
def relu_forward(x):
    # Cache for output
    cache = x

    # Implementing ReLU forward pass
    # Numbers that are less than zero will be changed to 0
    relu_output = np.maximum(0, x)

    # Returning calculated ReLU output
    return relu_output, cache
```

#### <a id="backward-pass-for-relu-layer">Backward Pass for ReLU layer</a>
Defining function for backward pass for ReLU layer.
```py
"""
Defining function for computing backward pass for ReLU layer.
ReLU layer - this is rectified linear units layer.
Input consists of following:
    derivatives_out - Upstream derivatives of any shape,
    cache - is x, of the same shape as derivatives_out.
Function returns:
    dx - gradient with respect to x.
"""
```

Consider following part of the code:
<br/>(related file: [layers.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Helper_Functions/layers.py))

```py
def relu_backward(derivatives_out, cache):
    # Preparing variable for input from cache
    x = cache

    # Implementing backward pass for ReLU layer
    # Creating array with the same shape as x
    # Filling with 'True' and 'False' according to the condition 'x > 0'
    temp = x > 0
    # Calculating gradient with respect to x
    dx = temp * derivatives_out

    # Backward pass for ReLU layer will return gradient with respect to x
    # Each element of the array will be filled with '0'
    # or derivative if that value in x was more than 0

    # Returning calculated ReLU output
    return dx
```

#### <a id="softmax-classification-loss">Softmax Classification loss</a>
Defining function for Softmax Classification loss.
```py
"""
Defining function for computing Logarithmic loss and gradient for Softmax Classification.
Input consists of following:
    x - input data of shape (N, C),
        where x[i, j] is score for the j-th class for the i-th input. 
    y - vector of labels of shape (N,),
        where y[i] is the label for x[i] and 0 <= y[i] < C.
Function returns:
    loss - scalar giving the Logarithmic loss,
    dx - gradient of loss with respect to x.
"""
```

Consider following part of the code:
<br/>(related file: [layers.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Helper_Functions/layers.py))

```py
def softmax_loss(x, y):
    # Calculating probabilities
    probabilities = np.exp(x - np.max(x, axis=1, keepdims=True))
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)

    # Getting number of samples
    N = x.shape[0]

    # Calculating Logarithmic loss
    loss = -np.sum(np.log(probabilities[np.arange(N), y])) / N

    # Calculating gradient
    dx = probabilities
    dx[np.arange(N), y] -= 1
    dx /= N

    # Returning tuple of Logarithmic loss and gradient
    return loss, dx
```

<br/>

### <a id="creating-classifier-model-of-cnn">Creating Classifier - model of CNN</a>
Creating model of CNN Classifier:
* Creating class for ConvNet1
* Initializing new Network
* Evaluating loss for training ConvNet1
* Calculating scores for predicting ConvNet1


#### <a id="initializing-new-network">Creating Class and Initializing new Network</a>
Consider following part of the code:
<br/>(related file: [ConvNet1.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Classifiers/ConvNet1.py))
```py
"""""""""
Initializing ConvNet1 with following architecture:
Conv - ReLU - Pooling - Affine - ReLU - Affine - Softmax
Neural Network operates on mini-batches of data of shape (N, C, H, W),
N is number of images, each with C channels, height H and width W.
"""

# Creating class for ConvNet1
class ConvNet1(object):

    """""""""
    Initializing new Network
    Input consists of following:
        input_dimension - tuple of shape (C, H, W) giving size of input data,
        number_of_filters - number of filters to use in Convolutional layer (which is only one here),
        size_of_filter - size of filter to use in the Convolutional layer (which is only one here),
        hidden_dimension - number of neurons to use in the Fully-Connected hidden layer,
        number_of_classes - number of scores to produce from the final Affine layer,
        weight_scale - scalar giving standard deviation for random initialization of weights,
        regularization - scala giving L2 regularization strength,
        dtype - numpy datatype to use for computation.
        
    """

    def __init__(self, input_dimension=(1, 28, 28), number_of_filters=32, size_of_filter=7,
                 hidden_dimension=100, number_of_classes=10, weight_scale=1e-3, regularization=0.0,
                 dtype=np.float32):

        # Defining dictionary to store all weights and biases
        self.params = {}
        # Defining variable for regularization
        self.regularization = regularization
        # Defining datatype for computation
        self.dtype = dtype
        # Getting input dimension C - channels, H - height, W - width
        C, H, W = input_dimension
        # Getting filter size which is squared
        HH = WW = size_of_filter
        # Getting number of filters
        F = number_of_filters
        # Getting number of neurons in hidden Affine layer
        Hh = hidden_dimension
        # Getting number of classes in output Affine layer
        Hclass = number_of_classes

        # Initializing weights and biases for Convolutional layer (which is only one here)
        # Weights are the volume of shape (F, C, HH, WW).
        # Where F is number of filters, each with C channels, height HH and width WW.
        # Biases initialized with 0 and shape (F,)
        self.params['w1'] = weight_scale * np.random.rand(F, C, HH, WW)
        self.params['b1'] = np.zeros(F)

        """
        Defining parameters for Convolutional layer (which is only one here):
            'cnn_params' is a dictionary with following keys:
                'stride' - step for sliding,
                'pad' - zero-pad frame around input that is calculated by following formula:
                    pad = (size_of_filter - 1) / 2
        
        Calculating spatial size of output image volume (feature maps) by following formulas:
            feature_maps - output data of feature maps of shape (N, F, Hc, Wc) where:
                Hc = 1 + (H + 2 * pad - HH) / stride
                Wc = 1 + (W + 2 * pad - WW) / stride
                    where,
                    N here is the same as we have it as number of input images,
                    F here is as number of channels of each N (that are now as feature maps),
                    HH and WW are height and width of filter.
        
        Input for CNN layer has shape of (N, C, H, W)
        Output from CNN layer has shape of (N, F, Hc, Wc)
        """

        self.cnn_params = {'stride': 1, 'pad': int((size_of_filter - 1) / 2)}
        Hc = int(1 + (H + 2 * self.cnn_params['pad'] - HH) / self.cnn_params['stride'])
        Wc = int(1 + (W + 2 * self.cnn_params['pad'] - WW) / self.cnn_params['stride'])

        """
        Defining parameters for Max Pooling layer:
            'pooling_params' is a dictionary with following keys:
                'pooling_height' - height of pooling region,
                'pooling_width' - width of pooling region,
                'stride' - step (distance) between pooling regions.
    
        Calculating spatial size of output image volume after Max Pooling layer
        by following formulas:
            output resulted data of shape (N, C, Hp, Wp) where:
                Hp = 1 + (Hc - pooling_height) / stride
                Wp = 1 + (Wc - pooling_width) / stride
                    where,
                    N here is the same as we have it as number of filters,
                    C here is as number of channels of each N,
                    Hc and Wc are height and width of output feature maps
                    from Convolutional layer.
                    
        Input for Max Pooling layer has shape of (N, F, Hc, Wc)
        Output from Max Pooling layer has shape of (N, F, Hp, Wp)
        """

        self.pooling_params = {'pooling_height': 2, 'pooling_width': 2, 'stride': 2}
        Hp = int(1 + (Hc - self.pooling_params['pooling_height']) / self.pooling_params['stride'])
        Wp = int(1 + (Wc - self.pooling_params['pooling_width']) / self.pooling_params['stride'])

        """
        Input for hidden Affine layer has shape of (N, F * Hp * Wp)
        Output from hidden Affine layer has shape of (N, Hh)
        """

        # Initializing weights and biases for Affine hidden layer
        # Weights are the volume of shape (F * Hp * Wp, Hh)
        # Where F * Hp * Wp performs full connections from Max Pooling layer to Affine hidden layer
        # Hh is number of neurons
        # Biases initialized with 0 and shape (Hh,)
        self.params['w2'] = weight_scale * np.random.rand(F * Hp * Wp, Hh)
        self.params['b2'] = np.zeros(Hh)

        """
        Input for output Affine layer has shape of (N, Hh)
        Output from output Affine layer has shape of (N, Hclass)
        """

        # Initializing weights and biases for output Affine layer
        # Weights are the volume of shape (Hh, Hclass)
        # Weights perform full connections from hidden to output layer
        # Hclass is number of neurons
        # Biases initialized with 0 and shape (Hh,)
        self.params['w3'] = weight_scale * np.random.rand(Hh, Hclass)
        self.params['b3'] = np.zeros(Hclass)

        # After initialization of Neural Network is done it is needed to set values as 'dtype'
        # Going through all keys from dictionary
        # Setting to all values needed 'dtype'

        for d_key, d_value in self.params.items():
            self.params[d_key] = d_value.astype(dtype)
```

#### <a id="evaluating-loss-for-training-convnet1">Evaluating loss for training ConvNet1</a>
Consider following part of the code:
<br/>(related file: [ConvNet1.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Classifiers/ConvNet1.py))
```py
 """
    Evaluating loss for training ConvNet1.
    Input consists of following:
        x of shape (N, C, H, W) - N data, each with C channels, height H and width W.
        y - vector of labels of shape (N,), where y[i] is the label for x[i].
    Function returns:      
        loss - scalar giving the Logarithmic loss,
        gradients - dictionary with the same keys as self.params,
                    mapping parameter names to gradients of loss
                    with respect to those parameters.
        
    """

    def loss_for_training(self, x, y):
        # Getting weights and biases
        w1, b1 = self.params['w1'], self.params['b1']
        w2, b2 = self.params['w2'], self.params['b2']
        w3, b3 = self.params['w3'], self.params['b3']

        # Implementing forward pass for ConvNet1 and computing class scores
        # Forward pass for Conv - ReLU - Pooling - Affine - ReLU - Affine
        cnn_output, cache_cnn = cnn_forward_naive(x, w1, b1, self.cnn_params)
        relu_output_1, cache_relu_1 = relu_forward(cnn_output)
        pooling_output, cache_pooling = max_pooling_forward_naive(relu_output_1, self.pooling_params)
        affine_hidden, cache_affine_hidden = affine_forward(pooling_output, w2, b2)
        relu_output_2, cache_relu_2 = relu_forward(affine_hidden)
        scores, cache_affine_output = affine_forward(relu_output_2, w3, b3)

        # Implementing backward pass for ConvNet1 and computing loss and gradients
        loss, d_scores = softmax_loss(scores, y)

        # Adding L2 regularization
        loss += 0.5 * self.regularization * np.sum(np.square(w1))
        loss += 0.5 * self.regularization * np.sum(np.square(w2))
        loss += 0.5 * self.regularization * np.sum(np.square(w3))

        # Backward pass through Affine output
        dx3, dw3, db3 = affine_backward(d_scores, cache_affine_output)
        # Adding L2 regularization
        dw3 += self.regularization * w3

        # Backward pass through ReLu and Affine hidden
        d_relu_2 = relu_backward(dx3, cache_relu_2)
        dx2, dw2, db2 = affine_backward(d_relu_2, cache_affine_hidden)
        # Adding L2 regularization
        dw2 += self.regularization * w2

        # Backward pass through Pooling, ReLu and Conv
        d_pooling = max_pooling_backward_naive(dx2, cache_pooling)
        d_relu_1 = relu_backward(d_pooling, cache_relu_1)
        dx1, dw1, db1 = cnn_backward_naive(d_relu_1, cache_cnn)
        # Adding L2 regularization
        dw1 += self.regularization * w1

        # Putting resulted derivatives into gradient dictionary
        gradients = dict()
        gradients['w1'] = dw1
        gradients['b1'] = db1
        gradients['w2'] = dw2
        gradients['b2'] = db2
        gradients['w3'] = dw3
        gradients['b3'] = db3

        # Returning loss and gradients
        return loss, gradients
```

#### <a id="calculating-scores-for-predicting-convnet1">Calculating scores for predicting ConvNet1</a>
Consider following part of the code:
<br/>(related file: [ConvNet1.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Classifiers/ConvNet1.py))
```py
"""
    Calculating scores for predicting ConvNet1.
    Input consists of following:
        x of shape (N, C, H, W) - N data, each with C channels, height H and width W.
    Function returns:
        scores - array of shape (N, C) giving classification scores,
                 where scores[i, C] is the classification score for x[i] and class C.
    """
    def scores_for_predicting(self, x):
        # Getting weights and biases
        w1, b1 = self.params['w1'], self.params['b1']
        w2, b2 = self.params['w2'], self.params['b2']
        w3, b3 = self.params['w3'], self.params['b3']

        # Implementing forward pass for ConvNet1 and computing class scores
        # Forward pass for Conv - ReLU - Pooling - Affine - ReLU - Affine
        cnn_output, _ = cnn_forward_naive(x, w1, b1, self.cnn_params)
        relu_output_1, _ = relu_forward(cnn_output)
        pooling_output, _ = max_pooling_forward_naive(relu_output_1, self.pooling_params)
        affine_hidden, _ = affine_forward(pooling_output, w2, b2)
        relu_output_2, _ = relu_forward(affine_hidden)
        scores, _ = affine_forward(relu_output_2, w3, b3)

        # Returning scores
        return scores
 ```

<br/>

### <a id="optimization-functions">Defining Functions for Optimization</a>
Using different types of optimization rules to update parameters of the Model.

#### <a id="vanilla-sgd">Vanilla SGD updating method</a>
Rule for updating parameters is as following:

![Vanilla SGD](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/vanilla_sgd.png)

Consider following part of the code:
<br/>(related file: [optimize_rules.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Helper_Functions/optimize_rules.py))
```py
# Creating function for parameters updating based on Vanilla SGD
def sgd(w, dw, config=None):
    # Checking if there was not passed any configuration
    # Then, creating config as dictionary
    if config is None:
        config = {}

    # Assigning to 'learning_rate' value by default
    # If 'learning_rate' was passed in config dictionary, then this will not influence
    config.setdefault('learning_rate', 1e-2)
	
    # Implementing update rule as Vanilla SGD
    w -= config['learning_rate'] * dw
	
    # Returning updated parameter and configuration
    return w, config
```

#### <a id="momentum-sgd">Momentum SGD updating method</a>
Rule for updating parameters is as following:

![Momentum SGD](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/momentum_sgd.png)

#### <a id="rms-propagation">RMS Propagation updating method</a>
Rule for updating parameters is as following:

![RMS](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/rms.png)

<br/>


### <a id="creating-solver-class">Creating Solver Class</a>
Creating Solver class for training classification models and for predicting:
* Creating and Initializing class for Solver
* Creating 'reset' function for defining variables for optimization
* Creating function 'step' for making single gradient update
* Creating function for checking accuracy of the model on the current provided data
* Creating function for training the model

Consider following part of the code:
<br/>(related file: [Solver.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Solver.py))
```py
# Creating class for Solver
class Solver(object):

    """""""""
    Initializing new Solver instance
    Input consists of following required and Optional arguments.
    
    Required arguments consist of following:
        model - a modal object conforming parameters as described above,
        data - a dictionary with training and validating data.
    
    Optional arguments (**kwargs) consist of following:
        update_rule - a string giving the name of an update rule in optimize_rules.py,
        optimization_config - a dictionary containing hyperparameters that will be passed 
                              to the chosen update rule. Each update rule requires different
                              parameters, but all update rules require a 'learning_rate' parameter.
        learning_rate_decay - a scalar for learning rate decay. After each epoch the 'learning_rate'
                              is multiplied by this value,
        batch_size - size of minibatches used to compute loss and gradients during training,
        number_of_epochs - the number of epoch to run for during training,
        print_every - integer number that corresponds to printing loss every 'print_every' iterations,
        verbose_mode - boolean that corresponds to condition whether to print details or not. 

    """

    def __init__(self, model, data, **kwargs):
        # Preparing required arguments
        self.model = model
        self.x_train = data['x_train']
        self.y_train = data['y_train']
        self.x_validation = data['x_validation']
        self.y_validation = data['y_validation']

        # Preparing optional arguments
        # Unpacking keywords of arguments
        # Using 'pop' method and setting at the same time default value
        self.update_rule = kwargs.pop('update_rule', 'sgd')  # Default is 'sgd'
        self.optimization_config = kwargs.pop('optimization_config', {})  # Default is '{}'
        self.learning_rate_decay = kwargs.pop('learning_rate_decay', 1.0)  # Default is '1.0'
        self.batch_size = kwargs.pop('batch_size', 100)  # Default is '100'
        self.number_of_epochs = kwargs.pop('number_of_epochs', 10)  # Default is '10'
        self.print_every = kwargs.pop('print_every', 10)  # Default is '10'
        self.verbose_mode = kwargs.pop('verbose_mode', True)  # Default is 'True'

        # Checking if there are extra keyword arguments and raising an error
        if len(kwargs) > 0:
            extra = ', '.join(k for k in kwargs.keys())
            raise ValueError('Extra argument:', extra)

        # Checking if update rule exists and raising an error if not
        # Using function 'hasattr(object, name)',
        # where 'object' is our imported module 'optimize_rules'
        # and 'name' is the name of the function inside
        if not hasattr(optimize_rules, self.update_rule):
            raise ValueError('Update rule', self.update_rule, 'does not exists')

        # Reassigning string 'self.update_rule' with the real function from 'optimize_rules'
        # Using function 'getattr(object, name)',
        # where 'object' is our imported module 'optimize_rules'
        # and 'name' is the name of the function inside
        self.update_rule = getattr(optimize_rules, self.update_rule)
```

#### <a id="reset">Defining function with additional variables</a>
Consider following part of the code:
<br/>(related file: [Solver.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Solver.py))
```py
    # Creating 'reset' function for defining variables for optimization
    def _reset(self):
        # Setting up variables
        self.current_epoch = 0
        self.best_validation_accuracy = 0
        self.best_params = {}
        self.loss_history = []
        self.train_accuracy_history = []
        self.validation_accuracy_history = []

        # Making deep copy of 'optimization_config' for every parameter at every layer
        # It means that at least learning rate will be for every parameter at every layer
        self.optimization_configurations = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optimization_config.items()}
            self.optimization_configurations[p] = d
```

#### <a id="step">Defining function for making single step</a>
Consider following part of the code:
<br/>(related file: [Solver.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Solver.py))
```py
    # Creating function 'step' for making single gradient update
    def _step(self):
        # Making minibatch from training data
        # Getting total number of training images
        number_of_training_images = self.x_train.shape[0]
        # Getting random batch of 'batch_size' size from total number of training images
        batch_mask = np.random.choice(number_of_training_images, self.batch_size)
        # Getting training dataset according to the 'batch_mask'
        x_batch = self.x_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # Calculating loss and gradient for current minibatch
        loss, gradient = self.model.loss_for_training(x_batch, y_batch)

        # Adding calculated loss to the history
        self.loss_history.append(loss)

        # Implementing updating for all parameters (weights and biases)
        # Going through all parameters
        for p, v in self.model.params.items():
            # Taking current value of derivative for current parameter
            dw = gradient[p]
            # Defining configuration for current parameter
            config_for_current_p = self.optimization_configurations[p]
            # Implementing updating and getting next values
            next_w, next_configuration = self.update_rule(v, dw, config_for_current_p)
            # Updating value in 'params'
            self.model.params[p] = next_w
            # Updating value in 'optimization_configurations'
            self.optimization_configurations[p] = next_configuration
```

#### <a id="accuracy">Defining function for checking accuracy</a>
Consider following part of the code:
<br/>(related file: [Solver.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Solver.py))
```py
    # Creating function for checking accuracy of the model on the current provided data
    # Accuracy will be used in 'train' function for both training dataset and for testing dataset
    # Depending on which input into the model will be provided
    def check_accuracy(self, x, y, number_of_samples=None, batch_size=100):

        """""""""
        Input consists of following:
            x of shape (N, C, H, W) - N data, each with C channels, height H and width W,
            y - vector of labels of shape (N,),
            number_of_samples - subsample data and test model only on this number of data,
            batch_size - split x and y into batches of this size to avoid using too much memory.

        Function returns:
            accuracy - scalar number giving percentage of images 
                       that were correctly classified by model.
        """

        # Getting number of input images
        N = x.shape[0]

        # Subsample data if 'number_of_samples' is not None
        # and number of input images is more than 'number_of_samples'
        if number_of_samples is not None and N > number_of_samples:
            # Getting random batch of 'number_of_samples' size from total number of input images
            batch_mask = np.random.choice(N, number_of_samples)
            # Reassigning (decreasing) N to 'number_of_samples'
            N = number_of_samples
            # Getting dataset for checking accuracy according to the 'batch_mask'
            x = x[batch_mask]
            y = y[batch_mask]

        # Defining and calculating number of batches
        # Also, making it as integer with 'int()'
        number_of_batches = int(N / batch_size)
        # Increasing number of batches if there is no exact match of input images over 'batch_size'
        if N % batch_size != 0:
            number_of_batches += 1

        # Defining variable for storing predicted class for appropriate input image
        y_predicted = []

        # Computing predictions in batches
        # Going through all batches defined by 'number_of_batches'
        for i in range(number_of_batches):
            # Defining start index and end index for current batch of images
            s = i * batch_size
            e = (i + 1) * batch_size
            # Getting scores by calling function 'loss_for predicting' from model
            scores = self.model.scores_for_predicting(x[s:e])
            # Appending result to the list 'y_predicted'
            # Scores is given for each image with 10 numbers of predictions for each class
            # Getting only one class for each image with maximum value
            y_predicted.append(np.argmax(scores, axis=1))
            # Example
            #
            # a = np.arange(6).reshape(2, 3)
            # print(a)
            #    ([[0, 1, 2],
            #     [3, 4, 5]])
            #
            # print(np.argmax(a))
            # 5
            #
            # np.argmax(a, axis=0)
            #     ([1, 1, 1])
            #
            # np.argmax(a, axis=1)
            #     ([2, 2])
            #
            # Now we have each image with its only one predicted class (index of each row)
            # but not with 10 numbers for each class

        # Concatenating list of lists and making it as numpy array
        y_predicted = np.hstack(y_predicted)

        # Finally, we compare predicted class with correct class for all input images
        # And calculating mean value among all values of following numpy array
        # By saying 'y_predicted == y' we create numpy array with True and False values
        # 'np.mean' function will return average of the array elements
        # The average is taken over the flattened array by default
        accuracy = np.mean(y_predicted == y)

        # Returning accuracy
        return accuracy
```

#### <a id="train">Defining function for running training procedure</a>
Consider following part of the code:
<br/>(related file: [Solver.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Solver.py))
```py
    # Creating function for training the model
    def train(self):
        # Getting total number of training images
        number_of_training_images = self.x_train.shape[0]
        # Calculating number of iterations per one epoch
        # If 'number_of_training_images' is less than 'self.batch_size' then we chose '1'
        iterations_per_one_epoch = max(number_of_training_images / self.batch_size, 1)
        # Calculating total number of iterations for all process of training
        # Also, making it as integer with 'int()'
        iterations_total = int(self.number_of_epochs * iterations_per_one_epoch)

        # Running training process in the loop for total number of iterations
        for t in range(iterations_total):
            # Making single step for updating all parameters
            self._step()

            # Checking if training loss has to be print every 'print_every' iteration
            if self.verbose_mode and t % self.print_every == 0:
                # Printing current iteration and showing total number of iterations
                # Printing currently saved loss from loss history
                print('Iteration: ' + str(t + 1) + '/' + str(iterations_total) + ',',
                      'loss =', self.loss_history[-1])

            # Defining variable for checking end of current epoch
            end_of_current_epoch = (t + 1) % iterations_per_one_epoch == 0

            # Checking if it is the end of current epoch
            if end_of_current_epoch:
                # Incrementing epoch counter
                self.current_epoch += 1
                # Decaying learning rate for every parameter at every layer
                for k in self.optimization_configurations:
                    self.optimization_configurations[k]['learning_rate'] *= self.learning_rate_decay

            # Defining variables for first and last iterations
            first_iteration = (t == 0)
            last_iteration = (t == iterations_total - 1)

            # Checking training and validation accuracy
            # At the first iteration, the last iteration, and at the end of every epoch
            if first_iteration or last_iteration or end_of_current_epoch:
                # Checking training accuracy with 1000 samples
                training_accuracy = self.check_accuracy(self.x_train, self.y_train,
                                                        number_of_samples=1000)

                # Checking validation accuracy
                # We don't specify number of samples as it has only 1000 images itself
                validation_accuracy = self.check_accuracy(self.x_validation, self.y_validation)

                # Adding calculated accuracy to the history
                self.train_accuracy_history.append(training_accuracy)
                self.validation_accuracy_history.append(validation_accuracy)

                # Checking if the 'verbose_mode' is 'True' then printing details
                if self.verbose_mode:
                    # Printing current epoch over total amount of epochs
                    # And training and validation accuracy
                    print('Epoch: ' + str(self.current_epoch) + '/' + str(self.number_of_epochs) + ',',
                          'Training accuracy = ' + str(training_accuracy) + ',',
                          'Validation accuracy = ' + str(validation_accuracy))

                # Tracking the best model parameters by comparing validation accuracy
                if validation_accuracy > self.best_validation_accuracy:
                    # Assigning current validation accuracy to the best validation accuracy
                    self.best_validation_accuracy = validation_accuracy
                    # Reset 'self.best_params' dictionary
                    self.best_params = {}
                    # Assigning current parameters to the best parameters variable
                    for k, v in self.model.params.items():
                        self.best_params[k] = v

        # At the end of training process swapping best parameters to the model
        self.model.params = self.best_params

        # Saving trained model parameters into 'pickle' file
        with open('Serialized_Models/model_params_ConvNet1.pickle', 'wb') as f:
            pickle.dump(self.model.params, f)

        # Saving loss, training accuracy and validation accuracy histories into 'pickle' file
        history_dictionary = {'loss_history': self.loss_history,
                              'train_accuracy_history': self.train_accuracy_history,
                              'validation_history': self.validation_accuracy_history}
        with open('Serialized_Models/model_histories_ConvNet1.pickle', 'wb') as f:
            pickle.dump(history_dictionary, f)
```

<br/>

### <a id="overfitting-small-data">Overfitting Small Data</a>
```py
import numpy as np

# Importing module 'ConvNet1.py'
from Helper_Functions.ConvNet1 import *

# Importing module 'Solver.py'
from Solver import *

# Loading data
# Opening file for reading in binary mode
with open('Data_Preprocessing/data.pickle', 'rb') as f:
    d = pickle.load(f, encoding='latin1')  # dictionary type

# Number of training examples
number_of_training_data = 100  # Can be changed and study with just 10 examples

# Preparing data by slicing in 'data' dictionary appropriate array
small_data = {
             'x_train':d['x_train'][:number_of_training_data],
             'y_train':d['y_train'][:number_of_training_data],
             'x_validation':d['x_validation'],
             'y_validation':d['y_validation']
             }

# Creating instance of class for 'ConvNet1' and initializing model
model = ConvNet1(input_dimension=(1, 28, 28), weight_scale=1e-2, hidden_dimension=100)

# Creating instance of class for 'Solver' and initializing model
solver = Solver(model,
                small_data,
                update_rule='sgd',
                optimization_config={'learning_rate':1e-3},
                learning_rate_decay=1.0,
                batch_size=50,
                number_of_epochs=50,  # Can be changed and study with just 40 epochs
                print_every=1,
                verbose_mode=True
               )

# Running training process
solver.train()
```

![Overfitting Small Data](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/overfitting_small_data_model_1_mnist.png)

<br/>

### <a id="training-results">Training Results</a>
```py
import numpy as np

# Importing module 'ConvNet1.py'
from Helper_Functions.ConvNet1 import *

# Importing module 'Solver.py'
from Solver import *

# Loading data
# Opening file for reading in binary mode
with open('Data_Preprocessing/data.pickle', 'rb') as f:
    d = pickle.load(f, encoding='latin1')  # dictionary type

# Creating instance of class for 'ConvNet1' and initializing model
model = ConvNet1(weight_scale=1e-3, hidden_dimension=500, regularization=1-e3)

# Creating instance of class for 'Solver' and initializing model
solver = Solver(model,
                d,
                update_rule='sgd',
                optimization_config={'learning_rate':1e-3},
                learning_rate_decay=1.0,
                batch_size=50,
                number_of_epochs=10,
                print_every=20,
                verbose_mode=True
               )

# Running training process
solver.train()
```

Training process of Model #1 with 12 000 iterations is shown on the figure below: 

![Training Model 1](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/training_model_1_mnist.png)

Initialized Filters and Trained Filters for ConvNet Layer is shown on the figure below:

![Filters Cifar10](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/filters_mnist.png)

Training process for Filters of ConvNet Layer is shown on the figure below:

![Training Filters Cifar10](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/mnist_filters_training.gif)


<br/>

### <a id="full-codes">Full codes are available here:</a>
* MNIST Digits Classification with `numpy` only:
  * `Data_Preprocessing`
    * `datasets`
    * [datasets_preparing.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Data_Preprocessing/datasets_preparing.py)
    * [mean_and_std.pickle](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Data_Preprocessing/mean_and_std.pickle)    
  * `Helper_Functions`
    * [layers.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Helper_Functions/layers.py)
    * [optimize_rules.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Helper_Functions/optimize_rules.py)
  * `Classifiers`
    * [ConvNet1.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Classifiers/ConvNet1.py) 
  * `Serialized_Models`
    * model1.pickle
  * [Solver.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Digits_Classification/Solver.py)

<br/>

### MIT License
### Copyright (c) 2018 Valentyn N Sichkar
### github.com/sichkar-valentyn
### Reference to:
Valentyn N Sichkar. Neural Networks for computer vision in autonomous vehicles and robotics // GitHub platform. DOI: 10.5281/zenodo.1317904
