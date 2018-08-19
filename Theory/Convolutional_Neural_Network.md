# Convolutional Neural Networks with Python
Convolutional Neural Networks in Python using <b>numpy</b> library.
<br/>[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1317904.svg)](https://doi.org/10.5281/zenodo.1317904)

## Content
Theory and experimental results (on this page):

* <a href="#Brief Introduction into Convolutional Neural Network">Brief Introduction into Convolutional Neural Network</a>
* <a href="#Task">Task</a>
* <a href="#Layers of CNN">Layers of CNN</a>
  * <a href="#Convolutional Layer">Convolutional Layer</a>
  * <a href="#Pooling Layer">Pooling Layer</a>
  * <a href="#ReLU Layer">ReLU Layer</a>
  * <a href="#Fully-Connected Layer">Fully-Connected Layer</a>
* <a href="#How does it work?">How does it work?</a>
* <a href="#Architecture of CNN">Architecture of CNN</a>

<br/>

### <a name="Brief Introduction into Convolutional Neural Network">Brief Introduction into Convolutional Neural Network</a>
**Definition**. **Convolutional Neural Network** (CNN, ConvNet) is a special architecture of artificial neural networks, aimed at effective image recognition, and it is a part of deep learning technologies. The working principle of **CNN** uses the features of the simple cells of the human visual cortex, responding to straight lines from different angles, as well as complex cells, whose reaction is associated with the activation of a certain set of simple cells. The idea of **CNN** is to alternate convolution layers and subsampling layers. The network structure is **feedforward** (without feedbacks), essentially multilayered. For training, standard methods are used, most often the method of **back propagation** of the error. The function of activation of neurons (transfer function) is any, at the choice of the researcher. The name of the network architecture is due to the existence of a convolution operation, the essence of which is that each fragment of the image is multiplied by the matrix (core) of convolution elementwise, and the result is summed and written to the same position of the output image.

**CNN** is very similar to conventional neural networks. They are also built on the basis of neurons that have learnable weights and biases. Each neuron receives some input data, performs a dot product of information and in some situations accompanies it by non-linearity. As in the case of conventional neural networks, the whole **CNN** expresses one differentiable score function: on the one hand it is raw pixels of the image, on the other - probability of the class or group of possible classes that characterize the picture.

But what is the difference then? The architecture of **CNN** makes an explicit assumption of the form "input data is images", which allows to encode certain properties into the architecture. Due to this feature, the preliminary announcement can be implemented more efficiently, while reducing the number of parameters in the network.

**Few words about architecture of CNN**. It is known that conventional neural networks receive input data (vector) and transform the information, passing it through a series of hidden layers. Each hidden layer consists of a set of neurons, where every neuron has connections with all neurons in the previous layer (fully connected) and where the neurons in the function of one layer are completely independent of each other and do not have common connections. The last fully connected layer is called the output layer, and in the classification settings it shows the class scores.

Conventional neural networks do not scale well in the case of large-sized images. **CNNs** use the fact that input data consist of images, and they restrict the network building in a more reasonable way. Unlike a conventional neural network, **CNN layers** consist of neurons located in 3 dimensions: **width, height and depth**, that is, dimensions that form the volume. For example, images on the input **CIFAR-10** are input volumes of activation, and the volume has dimensions of **32x32x3**, which is width, height and depth respectively. Neurons are connected only to a small area of the layer. In addition, the resulting output layer for this data set (CIFAR-10) will be **1x1x10**, since by the end of the **CNN architecture** the full image will be reduced into a single vector of class scores.

![Architecture of MLP and CNN](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/MLP_via_CNN.png)

As it is seen from the figure, every **CNN layer** converts 3D input volume into 3D output volume of neuron activations. Input layer contains an image with its width and height that are determined by the size of the picture, and the depth of 3 that are **red, green and blue channels**. The basis of the **CNNs** are the layers where each layer is characterized by a simple property - it converts the input data as a 3D volume into an output 3D volume with some differentiable function.

**CNN**s provide partial resistance to scale changes, offsets, turns, angles and other distortions. **CNN**s unite three architectural ideas to ensure invariance to scale, rotation and spatial distortion:
* local two-dimensional connectivity of neurons;
* common synaptic weights (provide detection of some features anywhere in the image and reduce the total number of weights);
* hierarchical organization with spatial subsamples.

At the moment, **CNN** and its modifications are considered the best in accuracy and speed algorithms for finding objects on the stage.

<br/>

### <a name="Task">Task</a>
The task of classifying images is the obtaining initial image as input and output its class (cat, dog, etc.) or a group of likely classes that best characterizes the image. When the computer gets the image (takes the input data), it sees an array of pixels. Depending on the resolution and size of the image, for example, the size of the array can be **32x32x3** (where 3 are the values of the **RGB channels**). Each of these numbers is assigned a value from **0 to 255**, which describes the intensity of the pixel at that point. These numbers are the only input data for the computer. The computer receives this matrix and displays numbers that describe the probability of the image class (**75%** for the cat, **20%** for the dog, **10%** for the bird, etc.).

<br/>

### <a name="Layers of CNN">Layers of CNN</a>
CNN is a sequence of layers. Each layer converts one volume of activations into another by means of a differentiable function. In the CNN, 3 main layers are used:
* **Convolutional Layer**
* **Pooling Layer (also known as subsampling layer)**
* **Fully Connected Layer**

These layers are used to build complete **CNN architecture**. A simple example of CNN for the task of classifying images using **CIFAR-10** data set can be the following architecture shown on the figure below.

![CNN architecture](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/CNN_architecture.png)

**Input (Input Layer)** - contains the original information about the image in the form of [32x32x3], where 32 is the width, another 32 is the height and 3 is the color channels - R, G, B (Red, Green and Blue).

**Conv (Conolutional Layer)** - is a set of maps (known also as feature maps), each of which has a filter (known also as core or kernel). The process of obtaining these maps can be described as following: the window of the filter size is going through whole image with a given step, and at each step the elemenrwise multiplication of values of local image window by filter is done, then result is summed (by adding all elements of the matrix all together) and written into the resulted output matrix of map. For example, if 12 filters are used, the output volume of the maps will be [32x32x12].

**ReLU (Rectified Linear Unit Layer)** - applies elementwise activation function (like f(x) = max(0, x)) with zero as threshold. In other words, it performs the following actions: if x > 0, then the value remains the same, and x < 0 changes this value by substituting it to 0.

**Pool (Pooling Layer)** - performs a downsampling operation of the spatial dimensions (width and height), as a result of which the volume can be reduced to [16×16×12]. At this stage, non-linear compaction of the feature maps is done. The logic of the process is as following: if some features have already been revealed in the previous convolution operation, then a detailed image is no longer needed for further processing, and it is compressed to less detailed image.

**FC (Fully-Connected Layer)** - displays a 10-dimensional vector of classes (as CIFAR-10 data set has 10 categories) to determine scores for each class. Each neuron is connected to all values in the previous volume.

Eventually, described CNN architecture, with its set of layers, converts an input image into an output vector with probability for every class. The image belongs to the class that obtain the biggest value.

<br/>

### <a name="Convolutional Layer">Convolutional Layer</a>
A convolutional layer is a set of **learnable filters** (known also as **core** or **kernel**) and with the help of which a set of **feature maps** (known also as **activation maps**) are obtained. Every filter produces its own feature map, or in other words - every feature map has its own filter. Consequently, convolutional layer consists of feature maps and corresponding filters.

The number of feature maps is determined by the requirements for the task. If we take a large number of maps, then the quality of recognition will increase, but the computational complexity will also increase. Analysis of scientific articles shows that it is recommended to take the ratio of one to two - each map of the previous layer is associated with two maps of the convolutional layer. For the first convolutional layer, the previous one is the input layer. If the input layer has three channels **R, G and B**, then each channel will be assosiated with two feature maps in convolutional layer, and first convolutional layer will have six feauture maps. The size of all maps in convolutional layer is the same and are calculated by the formula (although it can be different if there is a special rule to processes edges):

![Size_of_feature_map](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Size_of_feature_map.png)

where
<br/>**(width, height)** - is the size of obtained feature map,
<br/>**map_width** - is the width of previous map (or input layer if it is first convolutional layer),
<br/>**map_height** - is the height of previous map (or input layer if it is first convolutional layer),
<br/>**kernel_width** - is the width of the filter,
<br/>**kernel_height** - is the height of the filter.

Filter (or kernel) slides over the entire area of the previous map and finds certain features. For example, one filter could produce the largest signal in the area of eye, mouth, or nose during training process, and another filter might reveal other features. Filter size is usually taken in the range from 2x2 to 8x8. If filter size is small, then it will not be able to identify any feature, if it's too large, then the number of connections between neurons increases. One of the main characteristic of CNN is in the filters that have a system of **shared weights**. Common weights allow to reduce the number of connections between neurons (in contrast with typical multilayer network) and allow to find the same features across entire image area.

Filter for curve detection is shown below.

![Filter_for_curve_detection](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Filter_for_curve_detection.png)

Filter representation in pixels is shown below.

![Filter_representation_in_pixels](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Filter_representation_in_pixels.png)

Filter on the image is shown below.

![Filter_on_the_image](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Filter_on_the_image.jpg)

Filter implementation to the receptive field on the image by elementwise multiplication is shown below.

![Filter_implementation](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Filter_implementation.png)

Initially, values of each feature map in convolutional layer are equal to zero. Values of filter weights are randomly set in the range from -0.5 to 0.5 (if the special pre-trained filters are not used). Filter slides over the previous map (or over area on the input image if it is first convolutional layer) and performs a convolution operation. Mathematically it can be represented with equation:

![Convolution_operation](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Convolution_operation.png)

where
<br/>**f**  - is an initial matrix of input image,
<br/>**g** - is a filter (kernel) for convolution.

Convolution process is shown below.

![Convolution_process](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Convolution_process.png)

This operation can be described as follows: filter **g**, with its window of given size, slides over the whole image **f** with a given step (for example 1 or 2), then at each step elementwise multiplication process of two windows is done (filter window and appropriate image window), and result is summed and written into new matrix. Depending on the method of processing the edges of the original matrix, the resulted matrix may be less than the original, the same size or larger (in this example, obtained feature map is 3x3 size, but it can be 5x5 - the same with original input area, or even larger, or any other different size depending on the approach).

**Short summary about Convolutional Layer:**
* **Hyperparameters**:
  * number of filters (kernels) denoted as **K_number**,
  * size of filters (spatial dimension) denoted as **K_size**,
  * step for sliding (also known as stride) denoted as **Step**,
  * processing edges by zero-padding parameter denoted as **Pad**.
* Takes an input volume of size **Width_In × Height_In × Depth_In**.
* Gives an output volume of size **Width_Out × Height_Out × Depth_Out**, that are calculated by following equations:
  * **Width_Out = (Width_In - K_size + 2Pad) / Step + 1**,
  * **Height_Out = (Height_In - K_size + 2Pad) / Step + 1**,
  * **Depth_Out = K_number**.

General setting for hyperparameters are: **K_number = 2, K_size = 3, Step = 1, Pad = 1.**
<br/>Suppose that an input volume size is: **Width_In = 5, Height_In = 5, Depth_In = 3.**
<br/>Then it means that there are **two 3 × 3 filters**, and they are applied with **step 1**. As a result, output volume has a spatial dimension (width and height are equal) calculated with described above equation: **(5 - 3 + 2) / 1 + 1 = 5.**

<br/>

### <a name="Pooling Layer">Pooling Layer</a>
**Pooling Layer** (also known as **subsampling layer** or **downsampling layer**) is inserted between **Convolutional Layers** and aimed to reduce spatial dimension of feature maps (width and height) doing it separately for each map through depth of volume of the previous layer. When some features have already been identified in the previous convolution operation, then a detailed image is no longer needed for further processing, and it is compressed to less detailed. This operation also helps to control overfitting.

Pooling Layer usually has the most common filters with size 2x2 and step equal to 2. Every filter in Pooling Layer is doing **MAX operation** choosing maximum value from 4 numbers. As an output, there is the same amount of feature maps with its depth from previous Convolutional Layer but with downsampling spatial size 2 times (by width and height). An example is shown on the figure below.

![Pooling_process](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Pooling_process_with_MAX.png)

Apart from MAX operation, other functions can be applied, such as average pooling or normalization pooling, but they are used rarely.

**Hyperparameters**:
* size of filters (spatial dimension) denoted as **K_size**,
* step for sliding (also known as stride) denoted as **Step**,

Pooling layer takes an input volume of size **Width_In × Height_In × Depth_In** and gives an output volume of size **Width_Out × Height_Out × Depth_Out**, that are calculated by following equations:
* **Width_Out = (Width_In - K_size) / Step + 1**,
* **Height_Out = (Height_In - K_size) / Step + 1**,
* **Depth_Out = Depth_In**.

<br/>

### <a name="ReLU Layer">ReLU Layer</a>
One of the stages of Neural Network development is the choice of neuron activation function. The form of the **activation function** largely determines the functionality of the Neural Network and the method of its learning. The classic **Back Propagation** algorithm works well on two-layer and three-layer neural networks, but with further increase in depth, it becomes problematic. One of the reasons is the so-called attenuation of the gradients. As the error propagates from the output layer to the input layer on each layer, the current result is multiplied by the derivative of the activation function. The derivative of the traditional **sigmoid activation function** is less than unit, so after several layers the error will be close to zero. If, on the contrary, the activation function has an unbounded derivative (as, for example, a **hyperbolic tangent**), then an explosive increase in error can occur as it spreads, which leads to instability of the learning procedure. That is why **Convolutional Layers** use the **ReLU (Rectified Linear Unit)**, that represents a rectified linear activation function, and is expressed by the following formula:

![ReLU_activation_function](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/ReLU_activation_function.png)

Its essence lies in the fact that images become with **no negative values** - they are converted to 0.

The graph of the **ReLU function** is shown on the figure below:

![ReLU_activation_function_figure](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/ReLU_activation_function_figure.png)

**Advantages**:
* derivative of **ReLU function** is either unit or zero, and therefore no growth or attenuation of the gradients can occur. Multiplying unit by the delta of the error, we get the delta error. But if we used another function, for example, a **hyperbolic tangent**, then the delta error could either decrease, or increase. Hyperbolic tangent derivative returns a number with different sign and the magnitude that can greatly affect the attenuation or expansion of the gradient;

* calculation of sigmoid and hyperbolic tangent requires **large computational operations** such as exponentiation, while ReLU can be implemented using a simple threshold transformation of matrices,

* cuts unnecessary details for negative values in image matrices.

It can be noted that ReLU is not always reliable enough and in the process of learning it can fail for some neurons. For example, a large gradient passing through the ReLU can lead to an update of the weights that the given neuron is never activated again. If this happens, then, from now on, the gradient passing through this neuron will always be zero. Accordingly, this neuron will be disabled. For example, if the learning rate is too high, it may turn out that up to 50% of ReLUs will never be activated. This problem is solved by choosing the proper learning rate.

<br/>

### <a name="Fully-Connected Layer">Fully-Connected Layer</a>

### <a name="How does it work?">How does it work?</a>

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
