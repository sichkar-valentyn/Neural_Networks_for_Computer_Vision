# CIFAR-10 Image Classification with `numpy` only
Example on Image Classification with the help of CIFAR-10 dataset and Convolutional Neural Network.
<br/>[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1317904.svg)](https://doi.org/10.5281/zenodo.1317904)

## Test online [here](https://valentynsichkar.name/cifar10.html)

## Content
Theory and experimental results (on this page):

* [CIFAR-10 Image Classification with `numpy` only](#cifar10-image-classification-with-numpy-only)
  * [Loading batches of CIFAR-10 dataset](#loading-batches-of-cifar19-dataset)
  * [Plotting examples of images from CIFAR-10 dataset](#plotting-examples-of-images-from-cifar10-dataset)
  * [Preprocessing loaded CIFAR-10 dataset](#preprocessing-loaded-cifar10-dataset)
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
   * [Creating Solver Class](#creating-solver-class)
   * [Training Results](#training-results)

  
<br/>

### <a id="cifar10-image-classification-with-numpy-only">CIFAR-10 Image Classification with `numpy` only</a>
In this example we'll test CNN for Image Classification with the help of CIFAR-10 dataset.
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
* **Vanilla SGD** - Vanilla Stochastic Gradient Descent,
* **Momentum SGD** - Stochastic Gradient Descent with Momentum,
* **RMSProp** - Root Mean Square Propagation,
* **Adam** - Adaptive Moment Estimation,
* **SVM** - Support Vector Machine.

<br/>**For current example** following architecture will be used:
<br/>`Input` --> `Conv` --> `ReLU` --> `Pool` --> `Affine` --> `ReLU` --> `Affine` --> `Softmax`

<br/>**For current example** following parameters will be used:

| Parameter | Description |
| --- | --- |
| Weights Initialization | `HE Normal` |
| Weights Update Policy | `Adam` |
| Activation Functions | `ReLU` |
| Regularization | `L2` |
| Pooling | `Max` |
| Loss Functions | `Softmax` |

<br/>**File structure** with folders and functions can be seen on the figure below:

![Image_Classification_File_Structure.png](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Image_Classification_Files_Structure.png)

<br/>Also, **file structure** can be seen below:
* CIFAR-10 Image Classification with `numpy` only:
  * `Data_Preprocessing`
    * `datasets`
      * [get_CIFAR-10.sh](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Data_Preprocessing/datasets/get_CIFAR-10.sh)
    * [datasets_preparing.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Data_Preprocessing/datasets_preparing.py)
    * [mean_and_std.pickle](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Data_Preprocessing/mean_and_std.pickle)    
  * `Helper_Functions`
    * [layers.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Helper_Functions/layers.py)
    * optimize_rules.py
  * `Classifiers`
    * [ConvNet1.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Classifiers/ConvNet1.py) 
  * `Serialized_Models`
    * model1.pickle
  * Solver.py
  
<br/>

### <a id="loading-batches-of-cifar19-dataset">Loading batches of CIFAR-10 dataset</a>
First step is to prepare data from CIFAR-10 dataset.
<br/>Getting datasets CIFAR-10 under **Linux Ubuntu** by running file `get_CIFAR-10.sh`:
* From terminal moving to the directory `Image_Classification/datasets`
* Running file with following command: `./get_CIFAR-10.sh`
  * If there is an error that `permission denied` change permission by following command `sudo chmod +x get_CIFAR-10.sh`
  * And run again `./get_CIFAR-10.sh`

File will download archive from official resource, unzip archive and delete non-needed anymore archive.
<br/>As a result there has to appear new folder `cifar-10-batches-py` with following files:
* data_batch_1
* data_batch_2
* data_batch_3
* data_batch_4
* data_batch_5
* batches.meta
* test_batch

Writing code in Python.
<br/>Importing needed libraries.
<br/>Consider following part of the code:
<br/>(related file: [datasets_preparing.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Data_Preprocessing/datasets_preparing.py))

```py
"""Importing library for object serialization
which we'll use for saving and loading serialized models"""
import pickle

# Importing other standard libraries
import numpy as np
import os
import matplotlib.pyplot as plt
```

Creating function for loading single batch of CIFAR-10 dataset.
<br/>Consider following part of the code:
<br/>(related file: [datasets_preparing.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Data_Preprocessing/datasets_preparing.py))

```py
def single_batch_cifar10(file):
    # Opening file for reading in binary mode
    with open(file, 'rb') as f:
        d = pickle.load(f, encoding='latin1')  # dictionary type
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
```

Creating function for loading whole CIFAR-10 dataset.
<br/>Consider following part of the code:
<br/>(related file: [datasets_preparing.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Data_Preprocessing/datasets_preparing.py))

```py
def whole_cifar10():
    # Defining lists for adding all batch's data all together
    x_collect = []
    y_collect = []

    # Loading all 5 batches for training and appending them together
    for i in range(1, 6):
        # Preparing current filename
        filename = os.path.join('../datasets/cifar-10-batches-py', 'data_batch_' + str(i))
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
    filename = os.path.join('../datasets/cifar-10-batches-py', 'test_batch')
    x_test, y_test = single_batch_cifar10(filename)

    # Returning whole CIFAR-10 data for training and testing
    return x_train, y_train, x_test, y_test
```

<br/>

### <a id="plotting-examples-of-images-from-cifar10-dataset">Plotting examples of images from CIFAR-10 dataset</a>
After all batches were load and concatenated all together it is possible to show examples of training images.
<br/>Creating function for showing first 100 unique example of images from CIFAR-10 dataset.
<br/>Consider following part of the code:
<br/>(related file: [datasets_preparing.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Data_Preprocessing/datasets_preparing.py))

```py
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
```

For plotting images consider following part of the code:
<br/>(related file: [datasets_preparing.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Data_Preprocessing/datasets_preparing.py))

```py
# Plotting 100 examples of training images from 10 classes
x, y, x1, y1 = whole_cifar10()
plot_cifar10_examples(x.astype('int'), y.astype('int'))
```

Result can be seen on the image below.

![CIFAR-10_examples](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/CIFAR-10_examples.png)

<br/>

### <a id="preprocessing-loaded-cifar10-dataset">Preprocessing loaded CIFAR-10 dataset</a>
Next, creating function for preprocessing CIFAR-10 datasets for further use in classifier.
* Normalizing data by `dividing / 255.0` (!) - up to researcher
* Normalizing data by `subtracting mean image` and `dividing by standard deviation` (!) - up to researcher
* Transposing every dataset to make channels come first
* Returning result as dictionary

Consider following part of the code:
<br/>(related file: [datasets_preparing.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Data_Preprocessing/datasets_preparing.py))

```py
def pre_process_cifar10():
    # Loading whole CIFAR-10 datasets
    x_train, y_train, x_test, y_test = whole_cifar10()
    
    # Normalizing whole data by dividing /255.0
    # (!) Pay attention, that this step is up to researcher and can be omitted
    # In my case I study all possible options, however it takes time for training and analyzing
    x_train /= 255.0
    x_test /= 255.0

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

    # Normalizing data by subtracting mean image and dividing by standard deviation
    # Subtracting the dataset by mean image serves to center the data.
    # It helps for each feature to have a similar range and gradients don't go out of control.
    # Calculating mean image from training dataset along the rows by specifying 'axis=0'
    mean_image = np.mean(x_train, axis=0)  # numpy.ndarray (32, 32, 3)

    # Calculating standard deviation from training dataset along the rows by specifying 'axis=0'
    std = np.std(x_train, axis=0)  # numpy.ndarray (32, 32, 3)
    # Saving calculated 'mean_image' and 'std' into 'pickle' file
    # We will use them when preprocess input data for classifying
    # We will need to subtract and divide input image for classifying
    # As we're doing now for training, validation and testing data
    dictionary = {'mean_image': mean_image, 'std': std}
    with open('mean_and_std.pickle', 'wb') as f_mean_std:
        pickle.dump(dictionary, f_mean_std)
        
    # Subtracting calculated mean image from pre-processed datasets
    # (!) Pay attention, that this step is up to researcher and can be omitted
    x_train -= mean_image
    x_validation -= mean_image
    x_test -= mean_image
    
    # Dividing then every dataset by standard deviation
    # (!) Pay attention, that this step is up to researcher and can be omitted
    x_train /= std
    x_validation /= std
    x_test /= std

    # Transposing every dataset to make channels come first
    x_train = x_train.transpose(0, 3, 1, 2)  # (49000, 3, 32, 32)
    x_test = x_test.transpose(0, 3, 1, 2)  # (1000, 3, 32, 32)
    x_validation = x_validation.transpose(0, 3, 1, 2)  # (1000, 3, 32, 32)

    # Returning result as dictionary
    d = {'x_train': x_train, 'y_train': y_train,
         'x_validation': x_validation, 'y_validation': y_validation,
         'x_test': x_test, 'y_test': y_test}

    # Returning dictionary
    return d
```

After running created function, it is possible to see loaded, prepared and preprocessed CIFAR-10 datasets.
<br/>Consider following part of the code:
<br/>(related file: [datasets_preparing.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Data_Preprocessing/datasets_preparing.py))

```py
data = pre_process_cifar10()
for i, j in data.items():
    print(i + ':', j.shape)
```

As a result there will be following:
* `x_train: (49000, 3, 32, 32)`
* `y_train: (49000,)`
* `x_validation: (1000, 3, 32, 32)`
* `y_validation: (1000,)`
* `x_test: (1000, 3, 32, 32)`
* `y_test: (1000,)`

<br/>

### <a id="saving-and-loading-serialized-models">Saving and Loading serialized models</a>
Checking `pickle` library for saving and loading serialized models.
<br/>In order to test how it works, we'll save simple dictionary into file and will load it after.
<br/>Consider following part of the code:
<br/>(related file: [datasets_preparing.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Data_Preprocessing/datasets_preparing.py))

```py
# Writing dictionary into file in binary mode with 'pickle' library
# Defining simple dictionary for testing
d = {'data': 'image', 'class': 'cat'}
with open('test.pickle', 'wb') as f:
    pickle.dump(d, f)

# Opening file for reading in binary mode
with open('test.pickle', 'rb') as f:
    d = pickle.load(f, encoding='latin1')  # dictionary type, we use 'latin1' for python3
print(d)  # {'class': 'cat', 'data': 'image'}
```

Saving loaded, prepared and preprocessed CIFAR-10 datasets into `pickle` file.
<br/>Loading saved into file data and comparing if it is the same with original one.
<br/>Consider following part of the code:
<br/>(related file: [datasets_preparing.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Data_Preprocessing/datasets_preparing.py))

```py
# Saving loaded and preprocessed data into 'pickle' file
data = pre_process_cifar10()
with open('data.pickle', 'wb') as f:
    pickle.dump(data, f)

# Checking if preprocessed data is the same with saved data into file
# Opening file for reading in binary mode
with open('data.pickle', 'rb') as f:
    d = pickle.load(f, encoding='latin1')  # dictionary type

# Comparing if they are the same
print(np.array_equal(data['x_train'], d['x_train']))  # True
print(np.array_equal(data['y_train'], d['y_train']))  # True
print(np.array_equal(data['x_test'], d['x_test']))  # True
print(np.array_equal(data['y_test'], d['y_test']))  # True
print(np.array_equal(data['x_validation'], d['x_validation']))  # True
print(np.array_equal(data['y_validation'], d['y_validation']))  # True
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
<br/>(related file: [layers.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Helper_Functions/layers.py))

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
<br/>(related file: [layers.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Helper_Functions/layers.py))

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
<br/>(related file: [layers.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Helper_Functions/layers.py))

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
<br/>(related file: [layers.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Helper_Functions/layers.py))

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
<br/>(related file: [layers.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Helper_Functions/layers.py))

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
<br/>(related file: [layers.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Helper_Functions/layers.py))

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
<br/>(related file: [layers.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Helper_Functions/layers.py))

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
<br/>(related file: [layers.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Helper_Functions/layers.py))

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
<br/>(related file: [layers.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Helper_Functions/layers.py))

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
<br/>(related file: [ConvNet1.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Classifiers/ConvNet1.py))
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

    def __init__(self, input_dimension=(3, 32, 32), number_of_filters=32, size_of_filter=7,
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
<br/>(related file: [ConvNet1.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Classifiers/ConvNet1.py))
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
<br/>(related file: [ConvNet1.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Classifiers/ConvNet1.py))
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

### <a id="creating-solver-class">Creating Solver Class</a>


<br/>

### <a id="training-results">Training Results</a>

![Training Model 1](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/training_model_1.png)


<br/>

<br/>Full codes are available here:
* CIFAR-10 Image Classification with `numpy` only:
  * `Data_Preprocessing`
    * `datasets`
      * [get_CIFAR-10.sh](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Data_Preprocessing/datasets/get_CIFAR-10.sh)
    * [datasets_preparing.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Data_Preprocessing/datasets_preparing.py)
    * [mean_and_std.pickle](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Data_Preprocessing/mean_and_std.pickle)    
  * `Helper_Functions`
    * [layers.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Helper_Functions/layers.py)
    * optimize_rules.py
  * `Classifiers`
    * [ConvNet1.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Image_Classification/Classifiers/ConvNet1.py) 
  * `Serialized_Models`
    * model1.pickle
  * Solver.py

<br/>

### MIT License
### Copyright (c) 2018 Valentyn N Sichkar
### github.com/sichkar-valentyn
### Reference to:
Valentyn N Sichkar. Neural Networks for computer vision in autonomous vehicles and robotics // GitHub platform. DOI: 10.5281/zenodo.1317904
