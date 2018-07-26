# Backpropagation in Neural Network (NN) with Python
Explaining backpropagation on the three layer NN in Python using <b>numpy</b> library.
<br/>[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1317904.svg)](https://doi.org/10.5281/zenodo.1317904)

## Content
Theory and experimental results (on this page):

* <a href="#Three Layers NN">Three Layers NN</a>
* <a href="#Mathematical calculations">Mathematical calculations</a>
* <a href="#Backpropagation">Backpropagation</a>
* <a href="#Writing a code in Python">Writing a code in Python</a>

### <a name="Three Layers NN">Three Layers NN</a>
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

### <a name="Mathematical calculations">Mathematical calculations</a>
By using matrices it is possible to calculate the output for each set of inputs.
<br/>On the figure below operations between matrices are shown.

![Matrices_for_three_layers_NN.png](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/matrices_for_three_layers_NN.png)

<br/>First matrix corresponds to the inputs (Layer 0) and is multiplied by matrix of weights. As a result, the matrix with values for hidden layer (Layer 1) received which is further multiplied by another matrix of weights. And matrix with outputs (Layer 2) finally received.

### <a name="Backpropagation">Backpropagation</a>
Updating the weights is the process of adjusting or training the NN in order to get more accurate result. Backpropagation updates weights from last layer to the first layer.
* Firstly, the **error** for the output **Layer 2** is calculated, which is the difference between desired output and received output, and this is the error for the last output layer (Layer 2): <br/>**layer_2_error = Desired data - Received data**
* Secondly, the **delta** for the output **Layer 2** is calculated, which is used for correction the **Weights 1-2** and for finding the error for the hidden layer (Layer 1). The adjustments will be done in proportion to the value of error by using **Sigmoid Gradient** that was described in the [Introduction](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Theory/Introduction.md) part: <br/>**delta_2 = layer_2_error * layer_2 * (1 - layer_2)**
* Thirdly, the **error** for the hidden **Layer 1** is calculated, multiplying **delta_2** by **Weights 0-1**. In this way the comparison of influence of the weights from the first layer (Layer 0) to the weights from the hidden layer (Layer 1) is evaluated: <br/>**layer_1_error = delta_2 * weights_0-1**
* Finally, the **delta** for the hidden **Layer 1** is calculated for correction the **Weights 1-2**: <br/>**delta_1 = layer_1_error * layer_1 * (1 - layer_1)**

Just steps are shown below:
* **layer_2_error = Desired data - Received data**
* **delta_2 = layer_2_error * layer_2 * (1 - layer_2)**
* **layer_1_error = delta_2 * weights_0-1**
* **delta_1 = layer_1_error * layer_1 * (1 - layer_1)**

After the **delta**s for last (Layer 2) and hidden (Layer 1) layers were found, the weights are updated by multiplying matrices of outputs from layers on appropriate delta:
* **weights_1-2 += Layer_1 * delta_2**
* **weights_0-1 += Layer_0 * delta_1**

<br/>On the figure below appropriate matrices are shown.

![Matrices_for_three_layers_NN_1.png](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/matrices_for_three_layers_NN_1.png)

### <a name="Writing a code in Python">Writing a code in Python</a>
To write a code in Python for building and training three layers NN we will use <b>numpy</b> library to deal with matrices.

```py

# Importing 'numpy' library
import numpy as np


```



Full code is available here: [Code2_simple_three_layers_NN.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Code2_simple_three_layers_NN.py)

<br/>

### MIT License
### Copyright (c) 2018 Valentyn N Sichkar
### github.com/sichkar-valentyn
### Reference to:
Valentyn N Sichkar. Neural Networks for computer vision in autonomous vehicles and robotics // GitHub platform. DOI: 10.5281/zenodo.1317904
