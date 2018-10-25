# Backpropagation in Neural Network (NN) with Python
Explaining backpropagation on the three layer NN in Python using <b>numpy</b> library.
<br/>[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1317904.svg)](https://doi.org/10.5281/zenodo.1317904)

## Content
Theory and experimental results (on this page):

* [Three Layers NN](#three-layers-nn)
* [Mathematical calculations](#mathematical-calculations)
* [Backpropagation](#backpropagation)
* [Writing a code in Python](#writing-a-code-in-python)
* [Results](#results)
* [Analysis of results](#analysis-of-results)

<br/>

### <a id="three-layers-nn">Three Layers NN</a>
In order to solve more complex tasks, apart from that was described in the [Introduction](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Theory/Introduction.md) part, it is needed to use more layers in the NN. In this case the weights will be updated sequentially from the last layer to the input layer with respect to the confidance of the current results. This approach is called **Backpropagation**.
<br/><br/>Consider three layers NN.
<br/>On the figure below the NN is shown.
<br/>It has **Input layer** (Layer 0), **Hidden Layer** (Layer 1), **Output Layer** (Layer 2).
<br/>In the **Layer 0** there are three parameters to be considered, in the **Layer 1** there are four Hidden Neurons and in the **Layer 2** there is one Output Neuron.
<br/><br/>

![Three_layers_NN](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/three_layers_NN.png)

<br/>Consider the same inputs from [Introduction](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Theory/Introduction.md) part.
<br/><b>Training sets</b> and <b>Desired results</b> are shown below.
* Set of inputs #1 = 1 1 1; Output #1 = 1
* Set of inputs #2 = 1 0 1; Output #2 = 1
* Set of inputs #3 = 0 0 1; Output #3 = 0
* Set of inputs #4 = 0 1 1; Output #4 = 0

<b>Testing set</b> is shown below.
* Set of inputs #5 = 1 0 0</b>

**Weights 0-1** corresponds to the weights between Layer 0 and Layer 1.
<br/>**Weights 1-2** corresponds to the weights between Layer 1 and Layer 2.

<br/>

### <a id="mathematical-calculations">Mathematical calculations</a>
By using matrices it is possible to calculate the output for each set of inputs.
<br/>On the figure below operations between matrices are shown.

![Matrices_for_three_layers_NN.png](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/matrices_for_three_layers_NN.png)

<br/>First matrix corresponds to the inputs (Layer 0) and is multiplied by matrix of weights. As a result, the matrix with values for hidden layer (Layer 1) received which is further multiplied by another matrix of weights. And matrix with outputs (Layer 2) finally received.

<br/>

### <a id="backpropagation">Backpropagation</a>
Updating the weights is the process of adjusting or training the NN in order to get more accurate result. Backpropagation updates weights from last layer to the first layer.
* Firstly, the **error** for the output **Layer 2** is calculated, which is the difference between desired output and received output, and this is the error for the last output layer (Layer 2): <br/>**layer_2_error = Desired data - Received data**
* Secondly, the **delta** for the output **Layer 2** is calculated, which is used for correction the **Weights 1-2** and for finding the error for the hidden layer (Layer 1). The adjustments will be done in proportion to the value of error by using **Sigmoid Gradient** that was described in the [Introduction](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Theory/Introduction.md) part and with respect to [Gradient descent](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Theory/Gradient_descent.md): <br/>**delta_2 = layer_2_error * layer_2 * (1 - layer_2)**
* Thirdly, the **error** for the hidden **Layer 1** is calculated, multiplying **delta_2** by **Weights 1-2**. In this way the comparison of influence of the hidden layer (Layer 1) to the output error is evaluated: <br/>**layer_1_error = delta_2 * weights_1_2**
* Finally, the **delta** for the hidden **Layer 1** is calculated for correction the **Weights 1-2**: <br/>**delta_1 = layer_1_error * layer_1 * (1 - layer_1)**

Just steps are shown below:
* **layer_2_error = Desired data - Received data**
* **delta_2 = layer_2_error * layer_2 * (1 - layer_2)**
* **layer_1_error = delta_2 * weights_1_2**
* **delta_1 = layer_1_error * layer_1 * (1 - layer_1)**

After the **delta**s for last (Layer 2) and hidden (Layer 1) layers were found, the weights are updated by multiplying matrices of outputs from layers on appropriate delta:
* **weights_1_2 += Layer_1 * delta_2**
* **weights_0_1 += Layer_0 * delta_1**

<br/>On the figure below appropriate matrices are shown.

![Matrices_for_three_layers_NN_1.png](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/matrices_for_three_layers_NN_1.png)

<br/>

### <a id="writing-a-code-in-python">Writing a code in Python</a>
To write a code in Python for building and training three layers NN we will use <b>numpy</b> library to deal with matrices.

```py
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
```

<br/>

### <a id="results">Results</a>
Set of inputs and outputs for the training process:
<br/><b>input_set_for_training = np.array([[1, 1, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1]])</b>
<br/>The output set we transposes into the vector with function 'T':
<br/><b>output_set_for_training = np.array([[1, 1, 0, 0]]).T</b>

Weights of synapses initialized from the very beginning randomly
<br/><b>[[-0.16595599  0.44064899 -0.99977125 -0.39533485]
<br/>[-0.70648822 -0.81532281 -0.62747958 -0.30887855]
<br/>[-0.20646505  0.07763347 -0.16161097  0.370439  ]]</b>

Weights 1-2 between hidden layer (Layer 1) and output layer (Layer 2)
<br/><b>[[-0.5910955 ]
<br/>[ 0.75623487]
<br/>[-0.94522481]
<br/>[ 0.34093502]]</b>

The data for testing after training is [1, 0, 0] and the expected output is 1.
<br/>Result is:
<br/><b>[0.99619533]</b>
<br/> Congratulations! The output is equal to <b>0.99619533</b> which is very close to 1.

<br/>

### <a id="analysis-of-results">Analysis of results</a>
Analysing obtained results and building the figure with <b>Outputs</b> and number of <b>Iterations</b> in order to understand the raising accuracy of output and needed amount of iterations for training.
<br/>Consider following part of the code:

```py
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
```

As a result we get our figure:

![Figure_2](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/figure_2.png)

We can see that after <b>300 iterations</b> the accuracy doesn't change too much.
<br/>But how to calculate the exact number of iterations to achieve needed accuracy?
<br/>Consider final part of the code:

```py
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
```

As we can see, to reach the accuracy <b>0.99</b> we need <b>1010</b> iterations.

Full code is available here: [Intro_simple_three_layers_NN.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Intro_simple_three_layers_NN.py)

<br/>

### MIT License
### Copyright (c) 2018 Valentyn N Sichkar
### github.com/sichkar-valentyn
### Reference to:
Valentyn N Sichkar. Neural Networks for computer vision in autonomous vehicles and robotics // GitHub platform. DOI: 10.5281/zenodo.1317904
