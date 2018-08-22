# File: Intro_simple_three_layers_NN.py
# Description: Neural Networks for computer vision in autonomous vehicles and robotics
# Environment: PyCharm and Anaconda environment
#
# MIT License
# Copyright (c) 2018 Valentyn N Sichkar
# github.com/sichkar-valentyn
#
# Reference to:
# Valentyn N Sichkar. Neural Networks for computer vision in autonomous vehicles and robotics // GitHub platform. DOI: 10.5281/zenodo.1317904




# Creating a three layers NN by using mathematical 'numpy' library
# Using methods from the library to operate with matrices

# Importing 'numpy' library
import numpy as np
# Importing 'matplotlib' library to plot experimental results in form of figures
import matplotlib.pyplot as plt


# Creating a class for three layers Neural Network
class ThreeLayersNeuralNetwork():
    def __init__(self):
        # Using 'seed' for the random generator
        # It'll return the same random numbers each time the program runs
        # Very useful for debugging
        np.random.seed(1)

        # Modeling three layers Neural Network
        # Input layer (Layer 0) has three parameters
        # Hidden layer (Layer 1) has four neurons
        # Output layer (Layer 2) has one neuron
        # Initializing weights between input and hidden layers
        # The values of the weights are in range from -1 to 1
        # We receive matrix 3x4 of weights (3 inputs in Layer 0 and 4 neurons in Layer 1)
        self.weights_0_1 = 2 * np.random.random((3, 4)) - 1

        # Initializing weights between hidden and output layers
        # The values of the weights are in range from -1 to 1
        # We receive matrix 4x1 of weights (4 neurons in Layer 1 and 1 neuron in Layer 2)
        self.weights_1_2 = 2 * np.random.random((4, 1)) - 1

        # Creating a variable for storing matrix with output results
        self.layer_2 = np.array([])

    # Creating function for normalizing weights and other results by Sigmoid curve
    def normalizing_results(self, x):
        return 1 / (1 + np.exp(-x))

    # Creating function for calculating a derivative of Sigmoid function (gradient of Sigmoid curve)
    # Which is going to be used for back propagation - correction of the weights
    # This derivative shows how good is the current weights
    def derivative_of_sigmoid(self, x):
        return x * (1 - x)

    # Creating function for running and testing NN after training
    def run_nn(self, set_of_inputs):
        # Feed forward through three layers in NN
        # Results are returned in normalized form in appropriate dimensions
        layer_0 = set_of_inputs  # matrix 1x3
        layer_1 = self.normalizing_results(np.dot(layer_0, self.weights_0_1))  # matrix 1x3 * matrix 3x4 = matrix 1x4
        layer_2 = self.normalizing_results(np.dot(layer_1, self.weights_1_2))  # matrix 1x4 * matrix 4x1 = matrix 1x1
        return layer_2

    # Creating function for training the NN
    def training_process(self, set_of_inputs_for_training, set_of_outputs_for_training, iterations):
        # Training NN desired number of times
        for i in range(iterations):
            # Feeding our NN with training set and calculating output
            # Feed forward through three layers in NN
            # With 'numpy' library and function 'dot' we multiply matrices with values in layers to appropriate weights
            # Results are returned in normalized form in appropriate dimensions
            layer_0 = set_of_inputs_for_training  # matrix 4x3
            layer_1 = self.normalizing_results(np.dot(layer_0, self.weights_0_1))  # matrix 4x3 * matrix 3x4 = matrix 4x4
            self.layer_2 = self.normalizing_results(np.dot(layer_1, self.weights_1_2))  # matrix 4x4 * matrix 4x1 = matrix 4x1

            # Using Backpropagation for calculating values to correct weights
            # Calculating an error for output layer (Layer 2) which is the difference between desired output and obtained output
            # We subtract matrix 4x1 of received outputs from matrix 4x1 of desired outputs
            layer_2_error = set_of_outputs_for_training - self.layer_2

            # Showing the error each 500 iterations to track the improvements
            # Comment before analysis to prevent extra information to be shown
            if (i % 500) == 0:
                print('Final error after', i, 'iterations =', np.mean(np.abs(layer_2_error)))
                # Final error after 0 iterations = 0.4685343254580603
                # Final error after 500 iterations = 0.027359665117498422
                # Final error after 1000 iterations = 0.018014239352682853
                # Final error after 1500 iterations = 0.01424538015492187
                # Final error after 2000 iterations = 0.01209811882788332
                # Final error after 2500 iterations = 0.01067390131321088
                # Final error after 3000 iterations = 0.009643630560125674
                # Final error after 3500 iterations = 0.008855140776037976
                # Final error after 4000 iterations = 0.008227317734746435
                # Final error after 4500 iterations = 0.007712523196874705

            # Calculating delta for output layer (Layer 2)
            # Using sign '*' instead of function 'dot' of numpy library
            # In this way matrix 4x1 will be multiplied by matrix 4x1 element by element
            # For example,
            # n = np.array([[1], [1], [1], [1]])
            # m = np.array([[2], [3], [4], [5]])
            # n * m = [[2], [3], [4], [5]]
            # That is what we need now for this case
            delta_2 = layer_2_error * self.derivative_of_sigmoid(self.layer_2)

            # Calculating an error for hidden layer (Layer 1)
            # Multiplying delta_2 by weights between hidden layer and output layer
            # Shows us how much hidden layer (Layer 1) influences on to the layer_2_error
            layer_1_error = np.dot(delta_2, self.weights_1_2.T)

            # Calculating delta for hidden layer (Layer 1)
            delta_1 = layer_1_error * self.derivative_of_sigmoid(layer_1)

            # Implementing corrections of weights
            self.weights_1_2 += np.dot(layer_1.T, delta_2)
            self.weights_0_1 += np.dot(layer_0.T, delta_1)


# Creating three layers NN by initializing of instance of the class
three_layers_neural_network = ThreeLayersNeuralNetwork()

# Showing the weights of synapses initialized from the very beginning randomly
# Weights 0-1 between input layer (Layer 0) and hidden layer (Layer 1)
print('Weights 0-1')
print(three_layers_neural_network.weights_0_1)
print()
# [[-0.16595599  0.44064899 -0.99977125 -0.39533485]
#  [-0.70648822 -0.81532281 -0.62747958 -0.30887855]
#  [-0.20646505  0.07763347 -0.16161097  0.370439  ]]

# Weights 1-2 between hidden layer (Layer 1) and output layer (Layer 2)
print('Weights 1-2')
print(three_layers_neural_network.weights_1_2)
print()
# [[-0.5910955 ]
#  [ 0.75623487]
#  [-0.94522481]
#  [ 0.34093502]]

# Creating a set of inputs and outputs for the training process
# Using function 'array' of the 'numpy' library
# Creating matrix 4x3
input_set_for_training = np.array([[1, 1, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1]])
# Create matrix 1x4 and transpose it to matrix 4x1 at the same time
output_set_for_training = np.array([[1, 1, 0, 0]]).T

# Starting the training process with data above and number of repetitions of 5000
three_layers_neural_network.training_process(input_set_for_training, output_set_for_training, 5000)

# Showing the output results after training
print()
print('Output results after training:')
print(three_layers_neural_network.layer_2)
print()
# [[0.99181271]
#  [0.9926379 ]
#  [0.00744159]
#  [0.00613513]]

# After training process was finished and weights are adjusted we can test NN
# The data for testing is [1, 0, 0]
# The expected output is 1
print('Output result for testing data = ', three_layers_neural_network.run_nn(np.array([1, 0, 0])))

# Congratulations! The output is equal to 0.99619533 which is very close to 1
# [0.99619533]

# Analysis of results
# It is better to comment two lines in the class 'ThreeLayersNeuralNetwork' before analysis:
'''
if (i % 500) == 0:
    print('Final error after', i, 'iterations =', np.mean(np.abs(layer_2_error)))
'''
# In order not to show extra information

# Finding needed amount of iterations to reach curtain accuracy
# Creating list to store the resulting data
lst_result = []

# Creating list to store the number of iterations for each experiment
lst_iterations = []

# Creating a loop and collecting resulted output data with different numbers of iterations
# From 10 to 1000 with step=10
for i in range(10, 1000, 10):
    # Creating new instance of the NN class each time
    # In order not to be influenced from the previous training results
    three_layers_neural_network_analysis = ThreeLayersNeuralNetwork()

    # Starting the training process with number of repetitions equals to i
    three_layers_neural_network_analysis.training_process(input_set_for_training, output_set_for_training, i)

    # Collecting number of iterations in the list
    lst_iterations += [i]

    # Now we run trained NN with data for testing and obtain the result
    output = three_layers_neural_network_analysis.run_nn(np.array([1, 0, 0]))

    # Collecting resulted outputs in the list
    lst_result += [output]


# Plotting the results
plt.figure()
plt.plot(lst_iterations, lst_result, 'b')
plt.title('Iterations via Output')
plt.xlabel('Iterations')
plt.ylabel('Output')

# Showing the plot
plt.show()


# Find exact number of iterations in order to reach specific accuracy
i = 10  # Iterations
output = 0.1  # Output
# Creating a while loop and training NN
# Stop when output reached accuracy 0.99
while output < 0.99:
    # Creating new instance of the NN class each time
    # In order not to be influenced from the previous training results above
    three_layers_neural_network_analysis_1 = ThreeLayersNeuralNetwork()

    # Starting the training process with number of repetitions equals to i
    three_layers_neural_network_analysis_1.training_process(input_set_for_training, output_set_for_training, i)

    # Now we run trained NN with data for testing and obtain the result
    output = three_layers_neural_network_analysis_1.run_nn(np.array([1, 0, 0]))

    # Increasing the number of iterations
    i += 10


# Showing the found number of iterations for accuracy 0.99
print()
print('Number of iterations to reach accuracy 0.99 is', i)  # Needed numbers of iterations is equal to 1010

