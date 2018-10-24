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




# Creating Convolutional Neural Network Model


"""""""""
Initializing ConvNet1 with following architecture:
Conv - ReLU - Pooling - Affine - ReLU - Affine - Softmax

Neural Network operates on mini-batches of data of shape (N, C, H, W),
N is number of images, each with C channels, height H and width W.

"""

# Importing needed libraries
import numpy as np
# Importing needed modules (.py files)
from Image_Classification.Helper_Functions.layers import *


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

    """
    Evaluating loss for predicting ConvNet1.
    Input consists of following:
        x of shape (N, C, H, W) - N data, each with C channels, height H and width W.

    Function returns:
        scores - array of shape (N, C) giving classification scores,
                 where scores[i, C] is the classification score for x[i] and class C.

    """
    def loss_for_predicting(self, x):
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
