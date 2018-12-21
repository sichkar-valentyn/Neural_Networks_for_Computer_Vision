# MNIST Digits Classification with `numpy` only
Example on Digits Classification with the help of MNIST dataset and Convolutional Neural Network.
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
* **Vanilla SGD** - Vanilla Stochastic Gradient Descent,
* **Momentum SGD** - Stochastic Gradient Descent with Momentum,
* **RMSProp** - Root Mean Square Propagation,
* **Adam** - Adaptive Moment Estimation,
* **SVM** - Support Vector Machine.

<br/>**For current example** following architecture will be used:
<br/>`Input` --> `Conv` --> `ReLU` --> `Pool` --> `Affine` --> `ReLU` --> `Affine` --> `Softmax`

![Model_1_Architecture.png](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Model_1_Architecture_MNIST.png)

<br/>**For current example** following parameters will be used:

| Parameter | Description |
| --- | --- |
| Weights Initialization | `HE Normal` |
| Weights Update Policy | `Adam` |
| Activation Functions | `ReLU` |
| Regularization | `L2` |
| Pooling | `Max` |
| Loss Functions | `Softmax` |

<br/>

<br/>

<br/>

<br/>

<br/>

<br/>

<br/>

### MIT License
### Copyright (c) 2018 Valentyn N Sichkar
### github.com/sichkar-valentyn
### Reference to:
Valentyn N Sichkar. Neural Networks for computer vision in autonomous vehicles and robotics // GitHub platform. DOI: 10.5281/zenodo.1317904
