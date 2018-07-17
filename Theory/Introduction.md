# Introduction in Neural Network (NN) with Python
Explaining basic concepts on how NN works and implementing simple, kind of classical, task in Python using just <b>numpy</b> library without special toolkits and NN libraries.

### Basic concepts of artificial NN
In our brain there are billions of billions neurons that are connected with each other with so called synapses (connectors). When we are thinking, neurons send signals to another neurons and depending on the power of this signals collected by synapses, the neurons can be activated and produce output to another neurons.
<br/><br/> 
On the figure below the simple one neuron is shown.
<br/>It has three inputs and one output.

![Neuron](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/neuron.png)

<br/>Using this neuron we will solve the classical problem, that is shown below.
* Input #1 = 1 1 1; Output #1 = 1
* Input #2 = 1 0 1; Output #2 = 1
* Input #3 = 0 0 1; Output #3 = 0
* Input #4 = 0 1 1; Output #4 = 0

The Inputs above are called <b>Training sets</b> and the Outputs - <b>Desired results</b>.
<br/>We will try to find the output for the <b>Input #5 = 1 0 0</b> (the output should be equal to 1)
<br/>This fifth Input is called <b>Testing set</b>
<br/>In the mathematical representation of NN we use matrices with numbers and to operate with neurons we provide operations between these matrices.

### Training of the neuron
Process where we teach our neuron to produce desired results (to 'think') is called <b>Training process</b>.
<br/>In order to begin the training process we need firstly to give the synapses (our input lines) weights. These weights will influence the output of the neuron.
* So, we set the weights randomly, usually between 0 and 1.
* After that we can start the training process by collecting inputs and multiplying them by appropriate weights and sending further to the neuron.
* The neuron implements special mathematical equation to calculate the output.
* When the output is received the error is calculated. The error in this case is the difference between desired result and produced current output of the neuron.
* Using the error, the weights are adjusted according to the direction of this error with very small value.

The training steps described above we will repeat <b>5000 times</b>.
<br/>After the neuron was trained it got the ability to 'think' and will produce correct prediction with <b>Testing set</b>. This technique is called <b>Back propagation</b>.

Equation that is used by neuron for calculating the output is as following:

![Equation](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/equation_1.png)

where _w_ - is the weight and _I_ - is the Input value.

In order to normalize the output value, the Sigmoid function is used which is mathematically convenient in this case. The function is as following:

![Sigmoid Function](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/sigmoid_function.png)

Instead of the _x_ we put the sum from the previous equation and as a result will obtain the normalized output of the neuron in the range between _0_ and _1_. Till now we will not use threshold value to make this example simple.

Next step is to adjust the weights.
<br/>For calculation how much it is needed to change the weights we will use <b>Derivative</b> of the <b>Sigmoid equation</b> with respect to the output value of the neuron. There are few main reasons why to use derivative. The adjustments will be done in proportion to the value of error. As we multiply by input, which is _0_ or _1_, and if the input is _0_, the weight will not be changed. And if the current weight is pretty much correct, the <b>Sigmoid Gradient</b> will help not to change it too much. Sigmoid Gradient we find by taking the derivative from output of the neuron:

![Sigmoid Gradient](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/sigmoid_gradient.png)

where <b>O</b> - is an output of the neuron.
<br/>So, the finall equation to adjust weights is as following:

![Correct weights](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/correct_weights.png)

where <b>error</b> - is the difference between the desired output and neuron's output, <b>I</b> - is an input value, and <b>O</b> - is an output value.

### Writing a code in Python
To write a code in Python for building and training NN we will not use special toolkits or NN libraries. Instead we will use powerful <b>numpy</b> library to deal with matrices. Code with a lot of comments is shown below.

```py
import numpy 
```

### Results




## MIT License
## Copyright (c) 2018 Valentyn N Sichkar
## github.com/sichkar-valentyn
### Reference to:
[1] Valentyn N Sichkar. Neural Networks for computer vision in autonomous vehicles and robotics // GitHub platform [Electronic resource]. URL: https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision (date of access: XX.XX.XXXX)
