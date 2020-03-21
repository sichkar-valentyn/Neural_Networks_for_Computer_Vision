# Traffic Sign Classification with Convolutional Neural Networks in Python
Implementing Traffic Sign Classification in Python for Computer Vision tasks in Autonomous Vehicles.
<br/>[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1317904.svg)](https://doi.org/10.5281/zenodo.1317904)

## Test online [here](https://valentynsichkar.name/traffic_signs.html)

## :mortar_board: Related Course for Detection Tasks
**Training YOLO v3 for Objects Detection with Custom Data.** *Build your own detector by labelling, training and testing on image, video and in real time with camera.* Join here: [https://www.udemy.com/course/training-yolo-v3-for-objects-detection-with-custom-data/](https://www.udemy.com/course/training-yolo-v3-for-objects-detection-with-custom-data/?referralCode=A283956A57327E37DDAD)

Detections on video are shown below. **Trained weights** for detection tasks can be found in the course mentioned above.
![Detections on Video](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3400968%2F11bee8c0918c092b7d256b5254ba441c%2Fts_detections.gif?generation=1581794210627123&alt=media "Detections of Traffic Signs on Video")

Explore dataset used for training and detection tasks here: [https://www.kaggle.com/valentynsichkar/traffic-signs-dataset-in-yolo-format](https://www.kaggle.com/valentynsichkar/traffic-signs-dataset-in-yolo-format)

### :triangular_flag_on_post: Concept Map of the Course
![Concept Map of the Course](https://github.com/sichkar-valentyn/YOLO-v3-Objects-Detection-with-Custom-Data/blob/master/images/Concept_map_YOLO_3.png "Concept Map of the Course")

### :point_right: Join the Course
[https://www.udemy.com/course/training-yolo-v3-for-objects-detection-with-custom-data/](https://www.udemy.com/course/training-yolo-v3-for-objects-detection-with-custom-data/?referralCode=A283956A57327E37DDAD)

<br/>

## Content
Short description of the content. Full codes you can find inside the course by link above:

* [Loading Data](#loading-data)
  * [Unique Training Examples](#unique-examples)
  * [Histogram of Training Examples](#histogram-of-unique-examples)
  * [Good Quality Examples](#good-quality-examples)
  * [Table of Labels](#table-of-labels)
* [Preprocessing Data](#preprocessing-data)
* [Model 1](#model-1)
  * [Overfitting Small Data for Model 1](#overfitting-small-data-for-model-1)
  * [Training of Model 1](#training-of-model-1)
  * [Accuracy for different datasets](#accuracy-fo-different-datasets)
  * [Visualizing Filters for Model 1](#visualizing-filters)
  * [Visualizing Feature Maps for Model 1](#visualizing-feature-maps)
* [Predicting with image from test dataset](#predicting-with-image-from-test-dataset)
* [Predicting with user's image](#predicting-with-users-image)
* [Traffic Sign Classification in Real Time](#traffic-sign-classification-in-real-time)

<br/>

### <a id="loading-data">Loading Data</a>
Data used for this task is **German Traffic Sign Benchmarks (GTSRB)**.
<br/>Initially dataset consists of images in ***PPM*** format with different sizes. 

For current task datasets were organized in the same way as it was done for [CIFAR-10 Image Classification](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Theory/cifar10.md):
* **x_train, x_validation, x_test** - **4D** tensors as *numpy.ndarray* type with shapes *(12345, 3, 32, 32)*
* **y_train, y_validation, y_test** - **1D** tensors as *numpy.ndarray* type with shapes *(12345, )*

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
Examples of ***Unique Traffic Signs*** for every class from Training dataset are shown on the figure below.

![43 unique examples](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/43_unique_examples.png)

<br/>

### <a id="histogram-of-unique-examples">Histogram of Training Examples</a>
Histogram of 43 classes for training dataset with their number of examples for Traffic Signs Classification before and after **Equalization** by adding transformated images (brightness and rotation) from original dataset is shown on the figure below. After **Equalization**, training dataset has increased up to **86989 examples**.

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
<br/>Datasets **data0 - data3** have RGB images and datasets **data4 - data8** have Gray images.
  * data0.pickle - Shuffling
  * data1.pickle - Shuffling, /255.0 Normalization
  * data2.pickle - Shuffling, /255.0 + Mean Normalization
  * data3.pickle - Shuffling, /255.0 + Mean + STD Normalization
  * data4.pickle - Grayscale, Shuffling
  * data5.pickle - Grayscale, Shuffling, Local Histogram Equalization
  * data6.pickle - Grayscale, Shuffling, Local Histogram Equalization, /255.0 Normalization
  * data7.pickle - Grayscale, Shuffling, Local Histogram Equalization, /255.0 + Mean Normalization
  * data8.pickle - Grayscale, Shuffling, Local Histogram Equalization, /255.0 + Mean + STD Normalization
  
Shapes of data0 - data3 are as following (RGB):
* x_train: (86989, 3, 32, 32)
* y_train: (86989,)
* x_validation: (4410, 3, 32, 32)
* y_validation: (4410,)
* x_test: (12630, 3, 32, 32)
* y_test: (12630,)
 
Shapes of data4 - data8 are as following (Gray):
* x_train: (86989, 1, 32, 32)
* y_train: (86989,)
* x_validation: (4410, 1, 32, 32)
* y_validation: (4410,)
* x_test: (12630, 1, 32, 32)
* y_test: (12630,)

<br>Examples of some of them (`RGB`, `Gray`, `Local Histogram Equalization`) are shown on the figure below:

![Preprocessed_examples](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Preprocessed_examples.png)

<br/>

### <a id="model-1">Model 1</a>
For **Model 1** architecture will be used as it was done for [CIFAR-10 Image Classification](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/Theory/cifar10.md):
<br/>`Input` --> `Conv` --> `ReLU` --> `Pool` --> `Affine` --> `ReLU` --> `Affine` --> `Softmax`

<br/>For **Model 1** following parameters will be used:

| Parameter | Description |
| --- | --- |
| Weights Initialization | `HE Normal` |
| Weights Update Policy | `Adam` |
| Activation Functions | `ReLU` |
| Regularization | `L2` |
| Pooling | `Max` |
| Loss Functions | `Softmax` |

<br/>

### <a id="overfitting-small-data-for-model-1">Overfitting Small Data for Model 1</a>
For Overfitting Small Data of Model 1 dataset **'data8.pickle'** was chosen.
Overfitting Small Data with 10 training examples and 100 epochs is shown on the figure below.

![Overfitting_Small_Data_for_Model_1_TS.png](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Overfitting_Small_Data_for_Model_1_TS.png)

<br/>

### <a id="training-of-model-1">Training of Model 1</a>
For training Model 1 dataset **'data8.pickle'** was chosen as it reached the best accuracy over all datasets.
Model 1 with 'data8.pickle' dataset reached **0.989** training accuracy.
<br/>Training process of Model 1 with **17 500** iterations is shown on the figure below.

![Training_of_Model_1_TS.png](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Training_of_Model_1_TS.png)

<br/>

### <a id="accuracy-fo-different-datasets">Accuracy for different datasets</a>
Accuracy for different datasets is shown on the figure below.

![Accuracy_of_different_datasets_of_Model_1_TS.png](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Accuracy_of_different_datasets_of_Model_1_TS.png)

<br/>Table with training results is shown on the figure below.

| Dataset | Training Accuracy | Validation Accuracy |
| --- | --- | --- |
| data2 | `0.977` | `0.881` |
| data3 | `0.983` | `0.889` |
| data7 | `0.988` | `0.943` |
| data8 | `0.989` | `0.924` |

<br/>

### <a id="visualizing-filters">Visualizing Filters</a>
Initialized and Trained filters for CNN Layer are shown on the figure below.

![Filters_Initialized_TS.png](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Filters_Initialized_TS.png)

<br/>

### <a id="visualizing-feature-maps">Visualizing Feature Maps</a>
Feature maps of trained network for CNN Layer are shown on the figure below.

![Feature_Maps_CNN_TS.png](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Feature_Maps_CNN_TS.png)

<br/>

### <a id="predicting-with-image-from-test-dataset">Predicting with image from test dataset</a>
Prediction with image from Test Dataset **'data8.pickle'** is shown on the figure below.
<br/>Classified as **Speed limit (60km/h)**.

![Predicting_with_test_image.png](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Predicting_with_test_image.png)

<br/>

### <a id="predicting-with-users-image">Predicting with user's image</a>
Prediction with user's image is shown on the figure below.
<br/>Classified as **Keep right**.

![Predicting_with_users_image.png](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Predicting_with_users_image.png)

<br/>

### <a id="traffic-sign-classification-in-real-time">Traffic Sign Classification in Real Time</a>
Traffic Sign Classification with Convolutional Neural Network.
<br/>**Left:** Original frame with Detected Sign.
<br/>**Upper Right:** Cut frame with Detected Sign.
<br/>**Lower Right:** Classified frame by ConvNet according to the Detected Sign.

![Traffic_Sign_Classification_Small_Small.gif](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Traffic_Sign_Classification_Small_Small.gif)

<br/>

### MIT License
### Copyright (c) 2019 Valentyn N Sichkar
### github.com/sichkar-valentyn
### Reference to:
Valentyn N Sichkar. Neural Networks for computer vision in autonomous vehicles and robotics // GitHub platform. DOI: 10.5281/zenodo.1317904
