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




# Creating helper functions for checking gradients

import numpy as np


# Creating function that evaluate numeric gradient for a function that accepts a numpy array
# Function returns numpy array
def evaluate_numeric_gradient_array(f, x, df, h=1e-5):
    # Preparing variable for returning gradient
    gradient = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h
        positive = f(x).copy()
        x[ix] = old_value - h
        negative = f(x).copy()
        x[ix] = old_value

        gradient[ix] = np.sum((positive - negative) * df) / (2 * h)
        it.iternext()

    # Returning calculated gradient
    return gradient


# Creating function that evaluate numeric gradient of f at x
#   f should be a function that takes a single argument
#   x is the point (numpy array) to evaluate the gradient at
# Function returns gradient
def evaluate_numeric_gradient(f, x, verbose=True, h=1e-5):
    # fx = f(x)
    # Preparing variable for returning gradient
    gradient = np.zeros_like(x)

    # Iterating over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # Evaluating function at x+h
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h  # Incrementing by h
        positive = f(x)  # Evaluating f(x + h)
        x[ix] = old_value - h
        negative = f(x)  # Evaluating f(x - h)
        x[ix] = old_value  # Restoring

        # Calculating partial derivatives with centered formula
        gradient[ix] = (positive - negative) / (2 * h)  # Calculating slope

        # If verbose=True showing detailed results
        if verbose:
            print(ix, gradient[ix])

        # Step to next iteration
        it.iternext()

    # Returning calculated gradient
    return gradient
