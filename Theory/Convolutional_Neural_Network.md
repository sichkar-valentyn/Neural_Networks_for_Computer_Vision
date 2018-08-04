# Convolutional Neural Networks with Python
Convolutional Neural Networks in Python using <b>numpy</b> library.
<br/>[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1317904.svg)](https://doi.org/10.5281/zenodo.1317904)

## Content
Theory and experimental results (on this page):

* <a href="#Convolutional Neural Network">Convolutional Neural Network</a>
* <a href="#Layers of CNN">Layers of CNN</a>
  * <a href="#Convolutional Layer">Convolutional Layer</a>
  * <a href="#Pooling Layer">Pooling Layer</a>
  * <a href="#Normalization Layer">Normalization Layer</a>
  * <a href="#Fully-Connected Layer">Fully-Connected Layer</a>
* <a href="#Architecture of CNN">Architecture of CNN</a>


### <a name="Convolutional Neural Network">Convolutional Neural Network</a>
**Definition**
<br/>**Convolutional Neural Network** (CNN, ConvNet) is a special architecture of artificial neural networks, aimed at effective image recognition, and it is a part of deep learning technologies. The working principle of **CNN** uses the features of the simple cells of the human visual cortex, responding to straight lines from different angles, as well as complex cells, whose reaction is associated with the activation of a certain set of simple cells. The idea of **CNN** is to alternate convolution layers and subsampling layers. The network structure is **feedforward** (without feedbacks), essentially multilayered. For training, standard methods are used, most often the method of **back propagation** of the error. The function of activation of neurons (transfer function) is any, at the choice of the researcher. The name of the network architecture is due to the existence of a convolution operation, the essence of which is that each fragment of the image is multiplied by the matrix (core) of convolution elementwise, and the result is summed and written to the same position of the output image.

**CNN** is very similar to conventional neural networks. They are also built on the basis of neurons that have learnable weights and biases. Each neuron receives some input data, performs a dot product of information and in some situations accompanies it by non-linearity. As in the case of conventional neural networks, the whole **CNN** expresses one differentiable score function: on the one hand it is raw pixels of the image, on the other - probability of the class or group of possible classes that characterize the picture.

But what is the difference then? The architecture of **CNN** makes an explicit assumption of the form "input data is images", which allows to encode certain properties into the architecture. Due to this feature, the preliminary announcement can be implemented more efficiently, while reducing the number of parameters in the network.

**Few words about architecture of CNN**
It is known that conventional neural networks receive input data (vector) and transform the information, passing it through a series of hidden layers. Each hidden layer consists of a set of neurons, where every neuron has connections with all neurons in the previous layer (fully connected) and where the neurons in the function of one layer are completely independent of each other and do not have common connections. The last fully connected layer is called the output layer, and in the classification settings it shows the class scores.

Conventional neural networks do not scale well in the case of large-sized images. **CNNs** use the fact that input data consist of images, and they restrict the network building in a more reasonable way. Unlike a conventional neural network, **CNN layers** consist of neurons located in 3 dimensions: **width, height and depth**, that is, dimensions that form the volume. For example, images on the input **CIFAR-10** are input volumes of activation, and the volume has dimensions of **32x32x3**, which is width, height and depth respectively. Neurons are connected only to a small area of the layer. In addition, the resulting output layer for this data set (CIFAR-10) will be **1x1x10**, since by the end of the **CNN architecture** the full image will be reduced into a single vector of class scores.

![Architecture of MLP and CNN](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/MLP_via_CNN.png)

As it is seen from the figure, every **CNN layer** converts 3D input volume into 3D output volume of neuron activations. Input layer contains an image with its width and height that are determined by the size of the picture, and the depth of 3 that are **red, green and blue channels**. The basis of the **CNNs** are the layers where each layer is characterized by a simple property - it converts the input data as a 3D volume into an output 3D volume with some differentiable function.

**CNN**s provide partial resistance to scale changes, offsets, turns, angles and other distortions. **CNN**s unite three architectural ideas to ensure invariance to scale, rotation and spatial distortion:
* local two-dimensional connectivity of neurons;
* common synaptic weights (provide detection of some features anywhere in the image and reduce the total number of weights);
* hierarchical organization with spatial subsamples.

At the moment, **CNN** and its modifications are considered the best in accuracy and speed algorithms for finding objects on the stage.

**Task**
<br>The task of classifying images is the obtaining initial image as input and output its class (cat, dog, etc.) or a group of likely classes that best characterizes the image. When the computer gets the image (takes the input data), it sees an array of pixels. Depending on the resolution and size of the image, for example, the size of the array can be **32x32x3** (where 3 are the values of the **RGB channels**). Each of these numbers is assigned a value from **0 to 255**, which describes the intensity of the pixel at that point. These numbers are the only input data for the computer. The computer receives this matrix and displays numbers that describe the probability of the image class (**75%** for the cat, **20%** for the dog, **10%** for the bird, etc.).



### <a name="Layers of CNN">Layers of CNN</a>

### <a name="Convolutional Layer">Convolutional Layer</a>

### <a name="Pooling Layer">Pooling Layer</a>

### <a name="Normalization Layer">Normalization Layer</a>

### <a name="Fully-Connected Layer">Fully-Connected Layer</a>

### <a name="Architecture of CNN">Architecture of CNN</a>




```py
import numpy as np
```

Full code is available here: 

<br/>

### MIT License
### Copyright (c) 2018 Valentyn N Sichkar
### github.com/sichkar-valentyn
### Reference to:
Valentyn N Sichkar. Neural Networks for computer vision in autonomous vehicles and robotics // GitHub platform. DOI: 10.5281/zenodo.1317904
