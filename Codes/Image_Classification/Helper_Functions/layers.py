# File: layers.py
# Description: Neural Networks for computer vision in autonomous vehicles and robotics
# Environment: PyCharm and Anaconda environment
#
# MIT License
# Copyright (c) 2018 Valentyn N Sichkar
# github.com/sichkar-valentyn
#
# Reference to:
# Valentyn N Sichkar. Neural Networks for computer vision in autonomous vehicles and robotics // GitHub platform. DOI: 10.5281/zenodo.1317904




# Creating helper functions for dealing with CNN layers


import numpy as np


"""
Defining function for naive forward pass for convolutional layer.

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


"""
Defining function for computing forward pass for ReLU layer.
ReLU layer - this is rectified linear units layer.

Input consists of following:
    x - input data of any shape.

Function returns a tuple of:
    relu_output - output data of the same shape as x,
    cache - is x, needed in backward pass.

"""


def relu_forward(x):
    # Cache for output
    cache = x

    # Implementing ReLU forward pass
    # Numbers that are less than zero will be changed to 0
    relu_output = np.maximum(0, x)

    # Returning calculated ReLU output
    return relu_output, cache


"""
Defining function for computing backward pass for ReLU layer.
ReLU layer - this is rectified linear units layer.

Input consists of following:
    derivatives_out - Upstream derivatives of any shape,
    cache - is x, of the same shape as derivatives_out.

Function returns:
    dx - gradient with respect to x.

"""


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


def softmax_loss(x, y):
    # Calculating probabilities
    probabilities = np.exp(x - np.max(x, axis=1, keepdims=True))
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)

    # Getting number of samples
    N = x.shape[0]

    # Calculating Logarithmic loss
    loss = -np.sum(np.log(probabilities[np.arange(N), y])) / N

    # Calculating gradient
    dx = probabilities.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    # Returning tuple of Logarithmic loss and gradient
    return loss, dx

