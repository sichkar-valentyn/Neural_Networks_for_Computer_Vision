# MNIST Digits Classification with `numpy` only
Example on Digits Classification with the help of MNIST dataset of handwritten digits and Convolutional Neural Network.
<br/>[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1317904.svg)](https://doi.org/10.5281/zenodo.1317904)

## Test online [here](https://valentynsichkar.name/mnist.html)

## Content
Short description of the content. Full codes you can find inside the course by link above:

* [MNIST Digits Classification with `numpy` only](#mnist-digits-classification-with-numpy-library-only)
  * [Loading MNIST dataset](#loading-mnist-dataset)
  * [Plotting examples of digits from MNIST dataset](#plotting-examples-of-digits-from-mnist-dataset)
  * [Preprocessing loaded MNIST dataset](#preprocessing-loaded-mnist-dataset)
  * [Saving and Loading serialized models](#saving-and-loading-serialized-models)
  * [Functions for dealing with CNN layers](#functions-for-dealing-with-cnn-layers)
    * Naive Forward Pass for Convolutional layer
    * Naive Backward Pass for Convolutional layer
    * Naive Forward Pass for Max Pooling layer
    * Naive Backward Pass for Max Pooling layer
    * Forward Pass for Affine layer
    * Backward Pass for Affine layer
    * Forward Pass for ReLU layer
    * Backward Pass for ReLU layer
    * Softmax Classification loss
  * [Creating Classifier - model of CNN](#creating-classifier-model-of-cnn)
    * Initializing new Network
    * Evaluating loss for training ConvNet1
    * Calculating scores for predicting ConvNet1
  * [Functions for Optimization](#optimization-functions)
    * Vanilla SGD
  * [Creating Solver Class](#creating-solver-class)
    * _Reset
    * _Step
    * Checking Accuracy
    * Train
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

<br/>**For current example** following parameters will be used:

| Parameter | Description |
| --- | --- |
| Weights Initialization | `HE Normal` |
| Weights Update Policy | `Vanilla SGD` |
| Activation Functions | `ReLU` |
| Regularization | `L2` |
| Pooling | `Max` |
| Loss Functions | `Softmax` |

<br/>

### <a id="loading-mnist-dataset">Loading MNIST dataset</a>
After downloading files from official resource, there has to be following files:
* train-images-idx3-ubyte.gz
* train-labels-idx1-ubyte.gz
* t10k-images-idx3-ubyte.gz
* t10k-labels-idx1-ubyte.gz

<br/>

### <a id="plotting-examples-of-digits-from-mnist-dataset">Plotting examples of digits from MNIST dataset</a>
After dataset was load, it is possible to show examples of training images.
Result can be seen on the image below.

![MNIST_examples](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/MNIST_examples.png)

<br/>

### <a id="preprocessing-loaded-mnist-dataset">Preprocessing loaded MNIST dataset</a>
Next, creating function for preprocessing MNIST dataset for further use in classifier.
* Normalizing data by `dividing / 255.0` (!) - up to researcher
* Normalizing data by `subtracting mean image` and `dividing by standard deviation` (!) - up to researcher
* Transposing every dataset to make channels come first
* Returning result as dictionary

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

<br/>

### <a id="creating-classifier-model-of-cnn">Creating Classifier - model of CNN</a>
Creating model of CNN Classifier:
* Creating class for ConvNet1
* Initializing new Network
* Evaluating loss for training ConvNet1
* Calculating scores for predicting ConvNet1

<br/>

### <a id="optimization-functions">Defining Functions for Optimization</a>
Using different types of optimization rules to update parameters of the Model.

#### <a id="vanilla-sgd">Vanilla SGD updating method</a>
Rule for updating parameters is as following:

![Vanilla SGD](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/vanilla_sgd.png)

<br/>


### <a id="creating-solver-class">Creating Solver Class</a>
Creating Solver class for training classification models and for predicting:
* Creating and Initializing class for Solver
* Creating 'reset' function for defining variables for optimization
* Creating function 'step' for making single gradient update
* Creating function for checking accuracy of the model on the current provided data
* Creating function for training the model

<br/>

### <a id="overfitting-small-data">Overfitting Small Data</a>
Overfitting Small Data with 100 training examples and 50 epochs is show on the figure below.
![Overfitting Small Data](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/overfitting_small_data_model_1_mnist.png)

Overfitting Small Data with 10 training examples and 40 epochs is show on the figure below.
![Overfitting Small Data](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/overfitting_small_data_model_1_mnist_1.png)

<br/>

### <a id="training-results">Training Results</a>
Training process of Model #1 with 12 000 iterations is shown on the figure below: 

![Training Model 1](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/training_model_1_mnist.png)

Initialized Filters and Trained Filters for ConvNet Layer is shown on the figure below:

![Filters Cifar10](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/filters_mnist.png)

Training process for Filters of ConvNet Layer is shown on the figure below:

![Training Filters Cifar10](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/mnist_filters_training.gif)

<br/>

### MIT License
### Copyright (c) 2018 Valentyn N Sichkar
### github.com/sichkar-valentyn
### Reference to:
Valentyn N Sichkar. Neural Networks for computer vision in autonomous vehicles and robotics // GitHub platform. DOI: 10.5281/zenodo.1317904
