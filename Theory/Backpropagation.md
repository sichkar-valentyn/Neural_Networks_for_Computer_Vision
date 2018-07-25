# Backpropagation in Neural Network (NN) with Python
Explaining backpropagation on the three layer NN in Python using <b>numpy</b> library.
<br/>[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1317904.svg)](https://doi.org/10.5281/zenodo.1317904)

## Content
Theory and experimental results (on this page):

* <a href="#Concepts of backpropagation in NN">Concepts of backpropagation in NN</a>

### <a name="Concepts of backpropagation in NN">Concepts of backpropagation in NN</a>
In order to solve more complex tasks, that was described in the [Introduction](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Theory/Introduction.md) part, it is needed to use more layers in the NN. In this case the weights will be updated sequentially from the last layer to the input layer with respect to the confidance of the current results. <br/>Let's consider three layer NN.
<br/>On the figure below the NN is shown.
<br/>It has **Input layer** (Layer 0), **Hidden Layer** (Layer 1), **Output Layer** (Layer 2).
<br/>In the **Layer 0** there are three parameters to be considered, in the **Layer 1** there are four Hidden Neurons and in the **Layer 2** there is one Output Neuron.
<br/><br/>

![Three_layer_NN](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/three_layer_NN.png)

```py
Code
```



Full code is available here: [Code2_simple_three_layer_NN.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/Code2_simple_three_layer_NN.py)

<br/>

### MIT License
### Copyright (c) 2018 Valentyn N Sichkar
### github.com/sichkar-valentyn
### Reference to:
Valentyn N Sichkar. Neural Networks for computer vision in autonomous vehicles and robotics // GitHub platform. DOI: 10.5281/zenodo.1317904
