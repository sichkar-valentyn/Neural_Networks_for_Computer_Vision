# Traffic Sign Classification with Convolutional Neural Networks in Python
Implementing Traffic Sign Classification in Python for Computer Vision tasks.
<br/>[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1317904.svg)](https://doi.org/10.5281/zenodo.1317904)

## Content
Theory and experimental results (on this page):

* [Loading Data](#loading-data)

<br/>

### <a id="loading-data">Loading Data</a>
Data used for this task is **German Traffic Sign Benchmarks (GTSB)**.
<br>Initially datasets consist of images in ppm format with different sizes. 
<br>It is up to researcher how to prepare datasets from GTSB to feed Neural Network and can be done individually.

<br>For current task datasets were organized as it was done for [CIFAR-10 Image Classification](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Theory/cifar10.md):
* x_train, x_validation, x_test - 4D numpy.ndarray type with shapes (12345, 32, 32, 3)
* y_train, y_validation, y_test - 1D numpy.ndarray type with shapes (12345, )

Here, **12345** - number of examples, **32, 32, 3** - image with size of 32x32 and 3 channels.

<br>Examples of unique traffic signs for every class from training dataset is shown on the figure below.

![43 unique examples](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/43_unique_examples.png)

<br>Histogram of 43 classes with their number of examples is shown on the figure below.

![Histogram of 43 classes](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/histogram_of_43_classes.png)

<br>Examples of good quality traffic signs for every class to show in GUI for driver is shown on the figure below.

![43 good quality examples](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/43_good_quality_examples.png)






<br/>

### MIT License
### Copyright (c) 2019 Valentyn N Sichkar
### github.com/sichkar-valentyn
### Reference to:
Valentyn N Sichkar. Neural Networks for computer vision in autonomous vehicles and robotics // GitHub platform. DOI: 10.5281/zenodo.1317904
