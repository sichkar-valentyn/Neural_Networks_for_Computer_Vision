# Introduction in Neural Network (NN) with Python
Explaining basic concepts on how NN works and implementing simple, kind of classical, task in Python using just <b>numpy</b> library without special toolkits and without high-level NN libraries.
<br/>[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1317904.svg)](https://doi.org/10.5281/zenodo.1317904)

## Content
Theory and experimental results (on this page):

* [Basic concepts of artificial NN](#basic-concepts-of-artificial-nn)
* [Training of the neuron](#training-of-the-neuron)
* [Writing a code in Python](#writing-a-code-in-python)
* [Results](#results)
* [Analysis of results](#analysis-of-results)

<br/>

### <a id="basic-concepts-of-artificial-nn">Basic concepts of artificial NN</a>
In our brain there are billions of billions neurons that are connected with each other with so called synapses (connectors). When we are thinking, neurons send signals to another neurons and depending on the power of this signals collected by synapses, the neurons can be activated and produce output to another neurons.
<br/><br/> 
On the figure below the simple one neuron is shown.
<br/>It has three inputs and one output.

![Neuron](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/neuron.png)

<br/>Using this neuron we will solve the classical problem, that is shown below.
* Set of inputs #1 = 1 1 1; Output #1 = 1
* Set of inputs #2 = 1 0 1; Output #2 = 1
* Set of inputs #3 = 0 0 1; Output #3 = 0
* Set of inputs #4 = 0 1 1; Output #4 = 0

The Inputs above are called <b>Training sets</b> and the Outputs - <b>Desired results</b>.
<br/>We will try to find the output for the <b> Set of inputs #5 = 1 0 0</b> (the output should be equal to 1).
<br/>This fifth _set of inputs_ is called <b>Testing set</b>.
<br/>In the mathematical representation of NN we use matrices with numbers and to operate with neurons we provide operations between these matrices as it is shown below on the figure.

![Matrices](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/matrices.png) 

<br/>

### <a id="training-of-the-neuron">Training of the neuron</a>
Process where we teach our neuron to produce desired results (to 'think') is called <b>Training process</b>.
<br/>In order to begin the training process we need firstly to give the synapses (our input lines) weights. These weights will influence the output of the neuron.
* So, we set the weights randomly, usually between 0 and 1.
* After that we can start the training process by collecting inputs and multiplying them by appropriate weights and sending further to the neuron.
* The neuron implements special mathematical equation to calculate the output.
* When the output is received the error is calculated. The error in this case is the difference between desired result and produced current output of the neuron.
* Using the error, the weights are adjusted according to the direction of this error with very small value.

The training steps described above we will repeat <b>5000 times</b>.
<br/>After the neuron was trained it got the ability to 'think' and will produce correct prediction with <b>Testing set</b>. This technique is called <b>[Back propagation](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Theory/Backpropagation.md)</b>.

Equation that is used by neuron for calculating the output is as following:

![Equation](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/equation_1.png)

where _w_ - is the weight and _I_ - is the Input value.

In order to normalize the output value, the Sigmoid function is used which is mathematically convenient in this case. The function is as following:

![Sigmoid Function](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/sigmoid_function.png)

Instead of the _x_ we put the sum from the previous equation and as a result will obtain the normalized output of the neuron in the range between _0_ and _1_. Till now we will not use threshold value to make this example simple.

Next step is to adjust the weights.
<br/>For calculation how much it is needed to change the weights we will use <b>Derivative</b> of the <b>Sigmoid equation</b> with respect to the output value of the neuron. There are few main reasons why to use derivative. The adjustments will be done in proportion to the value of error. As we multiply by input, which is _0_ or _1_, and if the input is _0_, the weight will not be changed. And if the current weight is pretty much correct, the <b>Sigmoid Gradient</b> will help not to change it too much. Also, it is good to know [Gradient descent](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Theory/Gradient_descent.md). Sigmoid Gradient we find by taking the derivative from output of the neuron:

![Sigmoid Gradient](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/sigmoid_gradient.png)

where <b>O</b> - is an output of the neuron.
<br/>So, the finall equation to adjust weights is as following:

![Correct weights](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/correct_weights.png)

where <b>I</b> - is an input value, <b>error</b> - is the difference between the desired output and neuron's output, and <b>O</b> - is an output value.

<br/>

### <a id="writing-a-code-in-python">Writing a code in Python</a>
To write a code in Python for building and training NN we will not use special toolkits or NN libraries. Instead we will use powerful <b>numpy</b> library to deal with matrices. Code with a lot of comments is shown below.

```py
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
            # We multiply transposed matrix 4x3 of inputs (matrix.T = matrix 3x4) on matrix 4x1 for corrections
            # And receive matrix 3x1 of corrections
            corrections = np.dot(set_of_inputs_for_training.T, nn_error * self.derivative_of_sigmoid(nn_output))

            # Implementing corrections of weights
            # We add matrix 3x1 of current weights with matrix 3x1 of corrections and receive matrix 3x1 of new weights
            self.weights_of_synapses += corrections


# Creating NN by initializing instance of the class
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
 
```

<br/>

### <a id="results">Results</a>
Set of inputs and outputs for the training process:
<br/><b>input_set_for_training = np.array([[1, 1, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1]])</b>
<br/>The output set we transposes into the vector with function 'T':
<br/><b>output_set_for_training = np.array([[1, 1, 0, 0]]).T</b>

Weights of synapses initialized from the beginning randomly:
<br/><b>[[-0.16595599]</b>
<br/><b>[ 0.44064899]</b>
<br/><b>[-0.99977125]]</b>

Weights of synapses after training process:
<br/><b>[[ 8.95950703]</b>
<br/><b>[-0.20975775]</b>
<br/><b>[-4.27128529]]</b>

The data for testing after training is [1, 0, 0] and the expected output is 1.
<br/>Result is:
<br/><b>[0.99987151]</b>
<br/> Congratulations! The output is equal to <b>0.99987</b> which is very close to 1.

<br/>

### <a id="analysis-of-results">Analysis of results</a>
Now we're going to analyse obtained results and build the figure with <b>Outputs</b> and number of <b>Iterations</b> in order to understand the raising accuracy of output and needed amount of iterations for training.
<br/>Let's consider following part of the code:

```py
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

```

As a result we get our figure:

![Figure](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/figure_1.png)

We can see that after <b>200 iterations</b> the accuracy doesn't change too much.
<br/>But how to calculate the exact number of iterations to achieve needed accuracy?
<br/>Let's consider final part of the code:

```py
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

```

As we can see, to reach the accuracy <b>0.999</b> we need <b>740</b> iterations.

Full code is available here: [Intro_simple_NN.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Intro_simple_NN.py)

<br/>

### MIT License
### Copyright (c) 2018 Valentyn N Sichkar
### github.com/sichkar-valentyn
### Reference to:
Valentyn N Sichkar. Neural Networks for computer vision in autonomous vehicles and robotics // GitHub platform. DOI: 10.5281/zenodo.1317904
