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

<br/>The Inputs above are called <b>Training sets</b> and the Outputs - <b>Desired results</b>.
<br/>We will try to find the output for the <b>Input #5 = 1 0 0</b> (the output should be equal to 1)
<br/>This fifth Input is called <b>Testing set</b>
<br/>In the mathematical representation of NN we use matrices with numbers and to operate with neurons we provide operations between these matrices.

### Training of the neuron
Process where we teach our neuron to produce desired results (to 'think') is called <b>Training process</b>.
<br/>In order to begin the training process we need firstly to give the synapses (our input lines) weights. These weights will influence the output of the neuron. We set the weights randomly, usually between 0 and 1.
<br/>After that we can start the training process by collecting inputs and multiplying them by appropriate weights and sending further to the neuron.
<br/>The neuron implements special mathematical equation to calculate the output.
<br/>When the output is received the error is calculated. The error in this case is the difference between desired result and produced current output of the neuron.
<br/>Using the error the weights are adjusted according to the direction of this error with very small value.
<br/>The training steps described above we will repeat 5000 times.
<br/>After the neuron was trained it got the ability to 'think' and will produce correct prediction with <b>Testing set</b>. This technique is called <b>Back propagation</b>.



## MIT License
## Copyright (c) 2018 Valentyn N Sichkar
## github.com/sichkar-valentyn
### Reference to:
[1] Valentyn N Sichkar. Neural Networks for computer vision in autonomous vehicles and robotics // GitHub platform [Electronic resource]. URL: https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision (date of access: XX.XX.XXXX)
