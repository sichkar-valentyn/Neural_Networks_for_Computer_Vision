


# Importing needed library
import numpy as np


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
