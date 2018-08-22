# File: Intro_simple_NN.py
# Description: Neural Networks for computer vision in autonomous vehicles and robotics
# Environment: PyCharm and Anaconda environment
#
# MIT License
# Copyright (c) 2018 Valentyn N Sichkar
# github.com/sichkar-valentyn
#
# Reference to:
# Valentyn N Sichkar. Neural Networks for computer vision in autonomous vehicles and robotics // GitHub platform. DOI: 10.5281/zenodo.1317904




# Creating a simple NN by using mathematical 'numpy' library
# We will use several methods from the library to operate with matrices

# Importing 'numpy' library
import numpy as np
# Importing 'matplotlib' library to plot experimental results in form of figures
import matplotlib.pyplot as plt


# Creating a class for Neural Network
class NN():
    def __init__(self):
        # Using 'seed' for the random generator
        # It'll return the same random numbers each time the program runs
        np.random.seed(1)

        # Modeling simple Neural Network with just one neuron
        # Neuron has three inputs and one output
        # Initializing weights to 3 by 1 matrix
        # The values of the weights are in range from -1 to 1
        # We receive matrix 3x1 of weights (3 inputs and 1 output)
        self.weights_of_synapses = 2 * np.random.random((3, 1)) - 1

    # Creating function for normalizing weights and other results by Sigmoid curve
    def normalizing_results(self, x):
        return 1 / (1 + np.exp(-x))

    # Creating function for calculating a derivative of Sigmoid function (gradient of Sigmoid curve)
    # Which is going to be used for back propagation - correction of the weights
    # This derivative shows how good is the current weight
    def derivative_of_sigmoid(self, x):
        return x * (1 - x)

    # Creating function for running NN
    def run_nn(self, set_of_inputs):
        # Giving NN the set of input matrices
        # With 'numpy' function 'dot' we multiply set of input matrices to weights
        # Result is returned in normalized form
        # We multiply matrix 4x3 of inputs on matrix 3x1 of weights and receive matrix 4x1 of outputs
        return self.normalizing_results(np.dot(set_of_inputs, self.weights_of_synapses))

    # Creating function for training the NN
    def training_process(self, set_of_inputs_for_training, set_of_outputs_for_training, iterations):
        # Training NN desired number of times
        for i in range(iterations):
            # Feeding our NN with training set and calculating output
            # We multiply matrix 4x3 of inputs on matrix 3x1 of weights and receive matrix 4x1 of outputs
            nn_output = self.run_nn(set_of_inputs_for_training)

            # Calculating an error which is the difference between desired output and obtained output
            # We subtract matrix 4x1 of received outputs from matrix 4x1 of desired outputs
            nn_error = set_of_outputs_for_training - nn_output

            # Calculating correction values for weights
            # We multiply input to the error multiplied by Gradient of Sigmoid
            # In this way, the weights that do not fit too much will be corrected more
            # If some inputs are equal to 0, that will not influence to the value of weights
            # We use here function 'T' that transpose matrix and allows to multiply matrices
            # We multiply transposed matrix 4x3 of inputs (matrix.T = matrix 3x4) on matrix 4x1 of corrections
            # And receive matrix 3x1 of corrections
            corrections = np.dot(set_of_inputs_for_training.T, nn_error * self.derivative_of_sigmoid(nn_output))

            # Implementing corrections of weights
            # We add matrix 3x1 of current weights with matrix 3x1 of corrections and receive matrix 3x1 of new weights
            self.weights_of_synapses += corrections


# Creating NN by initializing of instance of the class
single_neuron_neural_network = NN()

# Showing the weights of synapses initialized from the very beginning randomly
# We create here matrix 3x1 of weights
print(single_neuron_neural_network.weights_of_synapses)
print()
# [[-0.16595599]
#  [ 0.44064899]
#  [-0.99977125]]

# Creating a set of inputs and outputs for the training process
# We use here function 'array' of the 'numpy' library
# We create here matrix 4x3
input_set_for_training = np.array([[1, 1, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1]])
# We create here matrix 1x4 and transpose it to matrix 4x1 at the same time
output_set_for_training = np.array([[1, 1, 0, 0]]).T

# Starting the training process with data above and number of repetitions of 5000
single_neuron_neural_network.training_process(input_set_for_training, output_set_for_training, 5000)

# Showing the weights of synapses after training process
print(single_neuron_neural_network.weights_of_synapses)
print()
# [[ 8.95950703]
#  [-0.20975775]
#  [-4.27128529]]

# After the training process was finished we can run our NN with data for testing and obtain the result
# The data for testing is [1, 0, 0]
# The expected output is 1
print(single_neuron_neural_network.run_nn(np.array([1, 0, 0])))

# Congratulations! The output is equal to 0.99987 which is very close to 1
# [0.99987151]


# Providing experimental analysis
# By this experiment we want to understand the needed amount of iterations to reach curtain accuracy
# Creating list to store the resulting data
lst_result = []

# Creating list to store the number of iterations for each experiment
lst_iterations = []

# Creating a loop and collecting resulted output data with different numbers of iterations
# From 10 to 1000 with step=10
for i in range(10, 1000, 10):
    # Create new instance of the NN class each time
    # In order not to be influenced from the previous training results
    single_neuron_neural_network_analysis = NN()

    # Starting the training process with number of repetitions equals to i
    single_neuron_neural_network_analysis.training_process(input_set_for_training, output_set_for_training, i)

    # Collecting number of iterations in the list
    lst_iterations += [i]

    # Now we run trained NN with data for testing and obtain the result
    output = single_neuron_neural_network_analysis.run_nn(np.array([1, 0, 0]))

    # Collecting resulted outputs in the list
    lst_result += [output]


# Plotting the results
plt.figure()
plt.plot(lst_iterations, lst_result, 'b')
plt.title('Iterations via Output')
plt.xlabel('Iterations')
plt.ylabel('Output')

# Showing the plots
plt.show()


# And finally lets find the exact number of needed iterations fo specific accuracy
i = 10  # Iterations
output = 0.1  # Output
# Creating a while loop and training NN
# Stop when output is with accuracy 0.999
while output < 0.999:
    # Again we create new instance of the NN class each time
    # In order not to be influenced from the previous training results above
    single_neuron_neural_network_analysis_1 = NN()

    # Starting the training process with number of repetitions equals to i
    single_neuron_neural_network_analysis_1.training_process(input_set_for_training, output_set_for_training, i)

    # Now we run trained NN with data for testing and obtain the result
    output = single_neuron_neural_network_analysis_1.run_nn(np.array([1, 0, 0]))

    # Increasing the number of iterations
    i += 10


# Showing the found number of iterations for accuracy 0.999
print(i)  # Needed numbers of iterations is equal to 740

