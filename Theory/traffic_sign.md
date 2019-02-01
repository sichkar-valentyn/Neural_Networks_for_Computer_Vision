# Traffic Sign Classification with Convolutional Neural Networks in Python
Implementing Traffic Sign Classification in Python for Computer Vision tasks in autonomous vehicles.
<br/>[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1317904.svg)](https://doi.org/10.5281/zenodo.1317904)

## Test online [here](https://valentynsichkar.name/traffic_signs.html)

## Content
Theory and experimental results (on this page):

* [Loading Data](#loading-data)
  * [Unique Training Examples](#unique-examples)
  * [Histogram of Training Examples](#histogram-of-unique-examples)
  * [Good Quality Examples](#good-quality-examples)
  * [Table of Labels](#table-of-labels)
* [Preprocessing Data](#preprocessing-data)
* [Model 1](#model-1)

<br/>

### <a id="loading-data">Loading Data</a>
Data used for this task is **German Traffic Sign Benchmarks (GTSB)**.
<br/>Initially datasets consist of images in *PPM* format with different sizes. 

For current task datasets were organized as it was done for [CIFAR-10 Image Classification](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Theory/cifar10.md):
* **x_train, x_validation, x_test** - 4D tensors as numpy.ndarray type with shapes *(12345, 3, 32, 32)*
* **y_train, y_validation, y_test** - 1D tensors as numpy.ndarray type with shapes *(12345, )*

Here,
<br/>**12345** - number of *images / labels*,
<br/>**3, 32, 32** - image with *3 channels* and size of *32x32 (height and width)*.

All tensors were put in a dictionary and were written in a pickle file:
<br/>
```py
d = {'x_train': x_train, 'y_train': y_train,
     'x_validation': x_validation, 'y_validation': y_validation,
     'x_test': x_test, 'y_test': y_test}
```

<br/>

### <a id="unique-examples">Unique Training Examples</a>
Examples of Unique Traffic Signs for every class from training dataset are shown on the figure below.

![43 unique examples](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/43_unique_examples.png)

<br/>

### <a id="histogram-of-unique-examples">Histogram of Training Examples</a>
Histogram of 43 classes for training dataset with their number of examples for Traffic Signs Classification before and after equalization by adding transformated images from original dataset is shown on the figure below.

![Histogram of 43 classes](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/histogram_of_43_classes.png)

<br/>

### <a id="good-quality-examples">Good Quality Examples</a>
Examples of Good Quality Traffic Signs for every class to show in GUI for driver are shown on the figure below.

![43 good quality examples](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/43_good_quality_examples.png)

<br/>

### <a id="table-of-labels">Table of Labels</a>
Following tabel represents number of class and its corresponding label (description).

| Class | Description |
| --- | --- |
| 0 | Speed limit (20km/h) |
| 1 | Speed limit (30km/h) |
| 2 | Speed limit (50km/h) |
| 3 | Speed limit (60km/h) |
| 4 | Speed limit (70km/h) |
| 5 | Speed limit (80km/h) |
| 6 | End of speed limit (80km/h) |
| 7 | Speed limit (100km/h) |
| 8 | Speed limit (120km/h) |
| 9 | No passing |
| 10 | No passing for vehicles over 3.5 metric tons |
| 11 | Right-of-way at the next intersection |
| 12 | Priority road |
| 13 | Yield |
| 14 | Stop |
| 15 | No vehicles |
| 16 | Vehicles over 3.5 metric tons prohibited |
| 17 | No entry |
| 18 | General caution |
| 19 | Dangerous curve to the left |
| 20 | Dangerous curve to the right |
| 21 | Double curve |
| 22 | Bumpy road |
| 23 | Slippery road |
| 24 | Road narrows on the right |
| 25 | Road work |
| 26 | Traffic signals |
| 27 | Pedestrians |
| 28 | Children crossing |
| 29 | Bicycles crossing |
| 30 | Beware of ice/snow |
| 31 | Wild animals crossing |
| 32 | End of all speed and passing limits |
| 33 | Turn right ahead |
| 34 | Turn left ahead |
| 35 | Ahead only |
| 36 | Go straight or right |
| 37 | Go straight or left |
| 38 | Keep right |
| 39 | Keep left |
| 40 | Roundabout mandatory |
| 41 | End of no passing |
| 42 | End of no passing by vehicles over 3.5 metric tons |

<br/>

### <a id="preprocessing-data">Preprocessing Data</a>
Prepared data is preprocessed in variety of ways and appropriate datasets are written into 'pickle' files.
  * data0.pickle - Shuffling
  * data1.pickle - Shuffling, /255.0 Normalization
  * data2.pickle - Shuffling, /255.0 + Mean Normalization
  * data3.pickle - Shuffling, /255.0 + Mean + STD Normalization
  * data4.pickle - Grayscale, Shuffling
  * data5.pickle - Grayscale, Shuffling, Local Histogram Equalization
  * data6.pickle - Grayscale, Shuffling, Local Histogram Equalization, /255.0 Normalization
  * data7.pickle - Grayscale, Shuffling, Local Histogram Equalization, /255.0 + Mean Normalization
  * data8.pickle - Grayscale, Shuffling, Local Histogram Equalization, /255.0 + Mean + STD Normalization 

<br>Examples of some of them (`RGB`, `Gray`, `Local Histogram Equalization`) are shown on the figure below:

![Preprocessed_examples](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Preprocessed_examples.png)

<br/>

### <a id="model-1">Model 1</a>
For **Model 1** architecture will be used as it was done for [CIFAR-10 Image Classification](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Theory/cifar10.md):
<br/>`Input` --> `Conv` --> `ReLU` --> `Pool` --> `Affine` --> `ReLU` --> `Affine` --> `Softmax`

![Model_1_Architecture_TS.png](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Model_1_Architecture_TS.png)

<br/>For **Model 1** following parameters will be used:

| Parameter | Description |
| --- | --- |
| Weights Initialization | `HE Normal` |
| Weights Update Policy | `Vanilla SGD` |
| Activation Functions | `ReLU` |
| Regularization | `L2` |
| Pooling | `Max` |
| Loss Functions | `Softmax` |

<br/>


### MIT License
### Copyright (c) 2019 Valentyn N Sichkar
### github.com/sichkar-valentyn
### Reference to:
Valentyn N Sichkar. Neural Networks for computer vision in autonomous vehicles and robotics // GitHub platform. DOI: 10.5281/zenodo.1317904
