# Convolutional Neural Networks with Python
Convolutional Neural Networks in Python using only pure `numpy` library.
<br/>[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1317904.svg)](https://doi.org/10.5281/zenodo.1317904)

## Content
Theory and experimental results (on this page):

* [Brief Introduction into Convolutional Neural Network](#brief-introduction-into-convolutional-neural-network)
* [Task](#task)
* [Layers of CNN](#layers-of-cnn)
  * [Convolutional Layer](#convolutional-layer)
  * [Pooling Layer](#pooling-layer)
  * [ReLU Layer](#relu-layer)
  * [Fully-Connected Layer](#fully-connected-layer)
* [Architecture of CNN](#architecture-of-cnn)
* [Video Summary for Introduction into CNN](#video-summary-for-introduction-into-cnn)
* [Writing code in Python](#writing-code-in-python)
  * [Simple Convolution with `numpy` only](#simple-convolution-with-numpy-only)
  * [More complex example with `numpy` only](#more-complex-example-with-numpy-only)

<br/>

### <a id="brief-introduction-into-convolutional-neural-network">Brief Introduction into Convolutional Neural Network</a>
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

### <a id="task">Task</a>
The task of classifying images is the obtaining initial image as input and output its class (cat, dog, etc.) or a group of likely classes that best characterizes the image. When the computer gets the image (takes the input data), it sees an array of pixels. Depending on the resolution and size of the image, for example, the size of the array can be **32x32x3** (where 3 are the values of the **RGB channels**). Each of these numbers is assigned a value from **0 to 255**, which describes the intensity of the pixel at that point. These numbers are the only input data for the computer. The computer receives this matrix and displays numbers that describe the probability of the image class (**75%** for the cat, **20%** for the dog, **10%** for the bird, etc.).

<br/>

### <a id="layers-of-cnn">Layers of CNN</a>
CNN is a sequence of layers. Each layer converts one volume of activations into another by means of a differentiable function. In the CNN, several main layers are used:
* **Input Layer**
* **Convolutional Layer**
* **ReLU Layer**
* **Pooling Layer (also known as subsampling layer)**
* **Fully Connected Layer**

These layers are used to build complete **CNN architecture**. A simple example of CNN for the task of classifying images using **CIFAR-10** data set can be the following architecture shown on the figure below.

![CNN architecture](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/CNN_architecture.png)

**Input (Input Layer)** - contains the original information about the image in the form of [32x32x3], where 32 is the width, another 32 is the height and 3 is the color channels - R, G, B (Red, Green and Blue).

**Conv (Convolutional Layer)** - is a set of maps (known also as feature maps), each of which has a filter (known also as core or kernel). The process of obtaining these maps can be described as following: the window of the filter size is going through whole image with a given step, and at each step the elementwise multiplication of values of local image window by filter is done, then result is summed (by adding all elements of the matrix all together) and written into the resulted output matrix of map. For example, if 12 filters are used, the output volume of the maps will be [32x32x12].

**ReLU (Rectified Linear Unit Layer)** - applies elementwise activation function (like f(x) = max(0, x)) with zero as threshold. In other words, it performs the following actions: if x > 0, then the value remains the same, and x < 0 changes this value by substituting it to 0.

**Pool (Pooling Layer)** - performs a downsampling operation of the spatial dimensions (width and height), as a result of which the volume can be reduced to [16×16×12]. At this stage, non-linear compaction of the feature maps is done. The logic of the process is as following: if some features have already been revealed in the previous convolution operation, then a detailed image is no longer needed for further processing, and it is compressed to less detailed image.

**FC (Fully-Connected Layer)** - displays a 10-dimensional vector of classes (as CIFAR-10 data set has 10 categories) to determine scores for each class. Each neuron is connected to all values in the previous volume.

Eventually, described CNN architecture, with its set of layers, converts an input image into an output vector with probability for every class. The image belongs to the class that obtain the biggest value.

<br/>

### <a id="convolutional-layer">Convolutional Layer</a>
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

### <a id="pooling-layer">Pooling Layer</a>
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

### <a id="relu-layer">ReLU Layer</a>
One of the stages of Neural Network development is the choice of neuron activation function. The form of the **activation function** largely determines the functionality of the Neural Network and the method of its learning. The classic **Back Propagation** algorithm works well on two-layer and three-layer neural networks, but with further increase in depth, it becomes problematic. One of the reasons is the so-called attenuation of the gradients. As the error propagates from the output layer to the input layer on each layer, the current result is multiplied by the derivative of the activation function. The derivative of the traditional **sigmoid activation function** is less than unit, so after several layers the error will be close to zero. If, on the contrary, the activation function has an unbounded derivative (as, for example, a **hyperbolic tangent**), then an explosive increase in error can occur as it spreads, which leads to instability of the learning procedure. That is why **Convolutional Layers** use the **ReLU (Rectified Linear Unit)**, that represents a rectified linear activation function, and is expressed by the following formula:

![ReLU_activation_function](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/ReLU_activation_function.png)

Its essence lies in the fact that images become with **no negative values** - they are converted to 0.

The graph of the **ReLU function** is shown on the figure below:

![ReLU_activation_function_figure](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/ReLU_activation_function_figure.png)

**Advantages**:
* derivative of **ReLU function** is either unit or zero, and therefore no growth or attenuation of the gradients can occur. Multiplying unit by the delta of the error, we get the delta error. But if we used another function, for example, a **hyperbolic tangent**, then the delta error could either decrease, or increase. Hyperbolic tangent derivative returns a number with different sign and the magnitude that can greatly affect the attenuation or expansion of the gradient;

* calculation of sigmoid and hyperbolic tangent requires **large computational operations** such as exponentiation, while ReLU can be implemented using a simple threshold transformation of matrices;

* cuts unnecessary details for negative values in image matrices.

It can be noted that ReLU is not always reliable enough and in the process of learning it can fail for some neurons. For example, a large gradient passing through the ReLU can lead to an update of the weights that the given neuron is never activated again. If this happens, then, from now on, the gradient passing through this neuron will always be zero. Accordingly, this neuron will be disabled. For example, if the learning rate is too high, it may turn out that up to 50% of ReLUs will never be activated. This problem is solved by choosing the proper learning rate.

<br/>

### <a id="fully-connected-layer">Fully-Connected Layer</a>
The last type of layers is **Fully Connected Layer**. Which is a conventional **Multilayer Perceptron**. Neurons in the last FC Layer have full connections with all the activations in the previous layer. The calculation of the neuron values in the FC Layer can be described by the formula:

![FC_neurons_value](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/FC_neurons_value.png)

where
<br/>**f()** - activation function;
<br/>**x** - feature map (activation map) ***j*** of layer ***l***;
<br/>**w** - weights of layer ***l***;
<br/>**b** - bias offset of layer ***l***.

After FC Layer, there is the last one - **Output Layer** of network, where **Softmax Function** is used to convert the outputs into probability values for each class as it is shown on the example on the figure below.

![Softmax_function](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Softmax_function.png)

<br/>

### <a id="architecture-of-cnn">Architecture of CNN</a>
The architecture of CNN is defined by the problem being solved. Below the typical architectures are shown.

![Architecture_of_CNN](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Architecture_of_CNN.png)

In the second example there is one **Conv layer** before every **Pooling layer**.
<br/>There are variety of different architectures that alternate main CNN layers between each other.

<br/>

### <a id="video-summary-for-introduction-into-cnn">Video Summary for Introduction into CNN</a>
Video Introduction into Convolutional NN with Python from scratch (summary):
<br/><a href="https://www.youtube.com/watch?v=04G3kRFI7pc" target="_blank"><img src="https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Video_Introduction_into_ConvNet.bmp" alt="Convolutional NN from scratch" /></a>

<br/>

### <a id="writing-code-in-python">Writing code in Python</a>
Experimental results on convolution applied to images with different filters.

### <a id="simple-convolution-with-numpy-only">Simple Convolution with `numpy` only</a>
Taking greyscale image and slicing it into the channels. Checking if all channels are identical.
<br/>Consider following part of the code:

```py
# Importing needed libraries
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Creating an array from image data
image_GreyScale = Image.open("images/owl_greyscale.jpg")
image_np = np.array(image_GreyScale)

# Checking the type of the array
print(type(image_np))  # <class 'numpy.ndarray'>
# Checking the shape of the array
print(image_np.shape)  # (1280, 830, 3)

# Showing image with every channel separately
channel_0 = image_np[:, :, 0]
channel_1 = image_np[:, :, 1]
channel_2 = image_np[:, :, 2]

# Checking if all channels are the same
print(np.array_equal(channel_0, channel_1))  # True
print(np.array_equal(channel_1, channel_2))  # True 
```

As it is seen, all three channels are identical as it is shown on the figure below.

![GreyScaled_image_with_three_identical_channels](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/GreyScaled_image_with_three_identical_channels.png)

For the further processing it is enough to work only with one channel.
<br/>In order to get **feature map** (convolved output image) in the same size, it is needed to set **Hyperparameters:**
* Filter (kernel) size, **K_size** = 3
* Step for sliding (stride), **Step** = 1
* Processing edges (zero valued frame around image), **Pad** = 1

Consequently, output image size is (width and height are the same):
* **Width_Out = (Width_In - K_size + 2 * Pad) / Step + 1**

Imagine, that input image is **5x5** spatial size (width and height), then output image:
* **Width_Out = (5 - 3 + 2 * 1)/1 + 1 = 5**, and this is equal to input image.

Taking so called **'identity'** filter and applying **convolution operation** with it to the one channel of input image.

![Identity_Filter](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Identity_Filter.png)

Consider following part of the code:

```py
# Taking as input image first channel as array
input_image = image_np[:, :, 0]
# Checking the shape
print(input_image.shape)  # (1080, 1920)

# Applying to the input image Pad frame with zero values
# Using NumPy method 'pad'
input_image_with_pad = np.pad(input_image, (1, 1), mode='constant', constant_values=0)
# Checking the shape
print(input_image_with_pad.shape)  # (1082, 1922)

# Defining so called 'identity' filter with size 3x3
# By applying this filter resulted convolved image has to be the same with input image
filter_0 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
# Checking the shape
print(filter_0.shape)  # (3, 3)

# Preparing zero valued output array for convolved image
# The shape is the same with input image according to the chosen Hyperparameters
output_image = np.zeros(input_image.shape)

# Implementing convolution operation
# Going through all input image with pad frame
for i in range(input_image_with_pad.shape[0] - 2):
    for j in range(input_image_with_pad.shape[1] - 2):
        # Extracting 3x3 patch (the same size with filter) from input image with pad frame
        patch_from_input_image = input_image_with_pad[i:i+3, j:j+3]
        # Applying elementwise multiplication and summation - this is convolution operation
        output_image[i, j] = np.sum(patch_from_input_image * filter_0)

# Checking if output image and input image are the same
# Because of the filter with only unit in the center (identity filter), convolution operation gives the same image
print(np.array_equal(input_image, output_image))  # True
```

As a result output image is identical to the input image, because of the **'identity' filter** that has the only unit in the middle.

Implementing another standard filters for edge detection.

![Filters_for_Edge_detection](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Filters_for_Edge_detection.png)

Consider following part of the code:

```py
# Defining standard filters (kernel) with size 3x3 for edge detection
filter_1 = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
filter_2 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
filter_3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
# Checking the shape
print(filter_1.shape, filter_2.shape, filter_3.shape)  # (3, 3) (3, 3) (3, 3)
```

In order to prevent appearing values that are more than 255 or less than 0, function is defined for corrections.
<br/>Consider following part of the code:

```py
# The following function is defined
def values_for_image_pixels(x_array):
    # Preparing resulted array
    result_array = np.zeros(x_array.shape)
    # Going through all elements of the given array
    for i in range(x_array.shape[0]):
        for j in range(x_array.shape[1]):
            # Checking if the element is in range [0, 255]
            if 0 <= x_array[i, j] <= 255:
                result_array[i, j] = x_array[i, j]
            elif x_array[i, j] < 0:
                result_array[i, j] = 0
            else:
                result_array[i, j] = 255
    # Returning edited array
    return result_array
```

Implementing convolution operations with three different filters separately.
<br/>Consider following part of the code:

```py
# Preparing zero valued output arrays for convolved images
# The shape is the same with input image according to the chosen Hyperparameters
output_image_1 = np.zeros(input_image.shape)
output_image_2 = np.zeros(input_image.shape)
output_image_3 = np.zeros(input_image.shape)

# Implementing convolution operation
# Going through all input image with pad frame
for i in range(input_image_with_pad.shape[0] - 2):
    for j in range(input_image_with_pad.shape[1] - 2):
        # Extracting 3x3 patch (the same size with filter) from input image with pad frame
        patch_from_input_image = input_image_with_pad[i:i+3, j:j+3]
        # Applying elementwise multiplication and summation - this is convolution operation
        # With filter_1
        output_image_1[i, j] = np.sum(patch_from_input_image * filter_1)
        # With filter_2
        output_image_2[i, j] = np.sum(patch_from_input_image * filter_2)
        # With filter_3
        output_image_3[i, j] = np.sum(patch_from_input_image * filter_3)


# Applying function to get rid of negative values and values that are more than 255
output_image_1 = values_for_image_pixels(output_image_1)
output_image_2 = values_for_image_pixels(output_image_2)
output_image_3 = values_for_image_pixels(output_image_3)
```

When convolution process is done, it is possible to see the results on the figures.

![Convolution_with_filters_for_edge_detection](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Convolution_with_filters_for_edge_detection.png)

Full code is available here: [CNN_Simple_Convolution.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/CNN_Simple_Convolution.py)

<br/>

### <a id="more-complex-example-with-numpy-only">More complex example with `numpy` only</a>
Consider more complex example of convolving input image with following architecture:
<br/>`Input` --> `Conv --> ReLU --> Pool` --> `Conv --> ReLU --> Pool` --> `Conv --> ReLU --> Pool`

**Hyperparameters** is as following:

* **Filter** (kernel) size, K_size = 3
* **Step** for sliding (stride), Step = 1
* **Processing edges** (zero valued frame around image), Pad = 1

Consequently, output image size is as following:
* **Width_Out** = (Width_In - K_size + 2 * Pad) / Step + 1
* **Height_Out** = (Height_In - K_size + 2 * Pad) / Step + 1

If an input image is 50x50 spatial size (width and height), then output image:
* Width_Out = Height_Out = (50 - 3 + 2 * 1)/1 + 1 = 50

Input image is **GrayScale** with three identical channels.
<br/>Preparing function for **2D Convolution** - just one image and one filter.
<br/>In this example **for** loops are used in order to deeply understand the process itself. But this approach is computationally expensive and in further examples **Fast Fourier Transform** will be used instead.  
<br/>Consider following part of the code:

```py
# Creating function for 2D convolution operation
def convolution_2d(image, filter, pad, step):
    # Size of the filter
    k_size = filter.shape[0]

    # Calculating spatial size - width and height
    width_out = int((image.shape[0] - k_size + 2 * pad) / step + 1)
    height_out = int((image.shape[1] - k_size + 2 * pad) / step + 1)

    # Preparing zero valued output array for convolved image
    output_image = np.zeros((width_out - 2 * pad, height_out - 2 * pad))

    # Implementing 2D convolution operation
    # Going through all input image
    for i in range(image.shape[0] - k_size + 1):
        for j in range(image.shape[1] - k_size + 1):
            # Extracting patch (the same size with filter) from input image
            patch_from_image = image[i:i+k_size, j:j+k_size]
            # Applying elementwise multiplication and summation - this is convolution operation
            output_image[i, j] = np.sum(patch_from_image * filter)

    # Returning result
    return output_image
```

Next, preparing function for **CNN Layer**.
<br/>Firstly, as input there is an image with three identical channels. That means every filter has to have three channels in depth also. If we consider second CNN Layer, then as input there is a set of feature maps produced by the first CNN Layer. It can be understood easier if we imagine that that set of feature maps is one image with its channels in depth. For example, first CNN Layer with four filters produces four feature maps that are input as one image with four channels for the second CNN Layer. Consequently, every filter for the second CNN Layer has to have four channels in depth also. Figure below shows process.

![Convolution_Process_P](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Convolution_Process_P.png)

Every filter with its channels in depth is convolved with input image (feature maps) with its depth appropriately. For example, first channel of the filter is convolving appropriate area in the first channel of input image, and second channel of the filter is convolving appropriate area (spatially the same as in the first channel) in the second channel of input image and so on. Result is summed up and written in appropriate cell of output feature map.

Consider following part of the code:

```py
# Creating function for CNN Layer
def cnn_layer(image_volume, filter, pad=1, step=1):
    # Note: image here can be a volume of feature maps, obtained in the previous layer

    # Applying to the input image volume Pad frame with zero values for all channels
    # Preparing zero valued array
    image = np.zeros((image_volume.shape[0] + 2 * pad, image_volume.shape[1] + 2 * pad, image_volume.shape[2]))

    # Going through all channels from input volume
    for p in range(image_volume.shape[2]):
        # Using NumPy method 'pad'
        # If Pad=0 the resulted image will be the same as input image
        image[:, :, p] = np.pad(image_volume[:, :, p], (pad, pad), mode='constant', constant_values=0)

    # Using following equations for calculating spatial size of output image volume:
    # Width_Out = (Width_In - K_size + 2*Pad) / Step + 1
    # Height_Out = (Height_In - K_size + 2*Pad) / Step + 1
    # Depth_Out = K_number
    # Size of the filter
    k_size = filter.shape[1]
    # Depth (number) of output feature maps - is the same with number of filters
    # Note: this depth will also be as number of channels for input image for the next layer
    depth_out = filter.shape[0]
    # Calculating spatial size - width and height
    width_out = int((image_volume.shape[0] - k_size + 2 * pad) / step + 1)
    height_out = int((image_volume.shape[1] - k_size + 2 * pad) / step + 1)

    # Creating zero valued array for output feature maps
    feature_maps = np.zeros((width_out, height_out, depth_out))  # has to be tuple with numbers

    # Implementing convolution of image with filters
    # Note: or convolving volume of feature maps, obtained in the previous layer, with new filters
    n_filters = filter.shape[0]

    # For every filter
    for i in range(n_filters):
        # Initializing convolved image
        convolved_image = np.zeros((width_out, height_out))  # has to be tuple with numbers

        # For every channel of the image
        # Note: or for every feature map from its volume, obtained in the previous layer
        for j in range(image.shape[-1]):
            # Convolving every channel (depth) of the image with every channel (depth) of the current filter
            # Result is summed up
            convolved_image += convolution_2d(image[:, :, j], filter[i, :, :, j], pad, step)
        # Writing results into current output feature map
        feature_maps[:, :, i] = convolved_image

    # Returning resulted feature maps array
    return feature_maps
```

Next, preparing function for that substitute pixel values that are more than 255.
<br/>Consider following part of the code:

```py
# Creating function for replacing pixel values that are more than 255 with 255
def image_pixels_255(maps):
    # Preparing array for output result
    r = np.zeros(maps.shape)
    # Replacing all elements that are more than 255 with 255
    # Going through all channels
    for c in range(r.shape[2]):
        # Going through all elements
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                # Checking if the element is less than 255
                if maps[i, j, c] <= 255:
                    r[i, j, c] = maps[i, j, c]
                else:
                    r[i, j, c] = 255
    # Returning resulted array
    return r
```

Next, preparing function for **ReLU Layer**. Here, all values that are negative is substituted with 0.
<br/>Consider following part of the code:

```py
# Creating function for ReLU Layer
def relu_layer(maps):
    # Preparing array for output result
    r = np.zeros_like(maps)
    # Using 'np.where' setting condition that every element in 'maps' has to be more than appropriate element in 'r'
    result = np.where(maps > r, maps, r)
    # Returning resulted array
    return result
```

Finally, preparing function for **Pooling Layer**. Obtained feature maps are downsampled in twice spatially with following parameters:
* **Size** of the filter is 2.
* **Step** for sliding is 2.

**MaxPooling** operation is implemented, that means that among four numbers (filter size 2x2) the maximum is chosen and is written in output feature map.
<br/>Consider following part of the code:

```py
# Creating function for Pooling Layer
def pooling_layer(maps, size=2, step=2):
    # Calculating spatial size of output resulted array - width and height
    # As our image has the same spatial size as input image (270, 480) according to the chosen Hyperparameters
    # Then we can use following equations
    width_out = int((maps.shape[0] - size) / step + 1)
    height_out = int((maps.shape[1] - size) / step + 1)

    # As filter size for pooling operation is 2x2 and step is 2
    # Then spatial size of pooling image will be twice less (135, 240)
    # Preparing zero valued output array for pooling image
    pooling_image = np.zeros((width_out, height_out, maps.shape[2]))

    # Implementing pooling operation
    # For all channels
    for c in range(maps.shape[2]):
        # Going through all image with step=2
        # Preparing indexes for pooling array
        ii = 0
        for i in range(0, maps.shape[0] - size + 1, step):
            # Preparing indexes for pooling array
            jj = 0
            for j in range(0, maps.shape[1] - size + 1, step):
                # Extracting patch (the same size with filter) from input image
                patch_from_image = maps[i:i+size, j:j+size, c]
                # Applying max pooling operation - choosing maximum element from the current patch
                pooling_image[ii, jj, c] = np.max(patch_from_image)
                # Increasing indexing for polling array
                jj += 1
            # Increasing indexing for polling array
            ii += 1

    # Returning resulted array
    return pooling_image
```

When following architecture **[ Conv - ReLU - Pool ] * 3** is implemented, it is possible to see the results on the figure.

![CNN_More_complex_example](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/CNN_More_complex_example.gif)

Full code is available here: [CNN_More_complex_example.py](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Codes/CNN_More_complex_example.py)

<br/>

### MIT License
### Copyright (c) 2018 Valentyn N Sichkar
### github.com/sichkar-valentyn
### Reference to:
Valentyn N Sichkar. Neural Networks for computer vision in autonomous vehicles and robotics // GitHub platform. DOI: 10.5281/zenodo.1317904
