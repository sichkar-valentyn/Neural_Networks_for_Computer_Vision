# Objects Detection Algorithms
Building Neural Networks for Objects Detection Algorithms.
<br/>**Under construction. Coming soon.**
<br/>[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1317904.svg)](https://doi.org/10.5281/zenodo.1317904)

## Content
Theory and experimental results (on this page):

* [Objects Detection Algorithms Overview](#main-objects-detection-algorithms)
  * [YOLO](#yolo)
  * [SSD](#ssd)
  * [DSSD](#dssd)
  * [R-FCN](#r-fcn)
  * [R-CNN](#r-cnn)
  * [Fast R-CNN](#fast-r-cnn)
  * [Faster R-CNN](#faster-r-cnn)
  * [Mask R-CNN](#mask-r-cnn)
  * [NASNet](#nasnet)
  * [FPN FRCN](#fpn-frcn)
  * [Retinanet](#retinanet)

<br/>

### <a id="main-objects-detection-algorithms">Objects Detection Algorithms Overview</a>
There are variety of algorithms for Detection Objects on the image. Fig.1.1 below shows the most popular of them. We will move step by step exploring each of them.

| ![Objects Detection Algorithms](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Objects_Detection/Objects_Detection_Algorithms___.png) | 
|:--:| 
| *Figure 1.1. Objects Detection Algorithms* |

<br/>

### <a id="yolo">YOLO</a>
**YOLO** stands for **You Only Look Ones**. Here we will describe how to prepare data in *YOLO* format, install *Darknet* framework, train *YOLO* and use it for *Objects Detection*.

Before starting to train *YOLO* it is needed to prepare data. It is possible to collect data or to use already prepared data.

| ![Data_Preparation](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/Objects_Detection/Data_Preparation.png) | 
|:--:| 
| *Figure 1.2. Data Preparation* |

Data has to be converted in *YOLO format*.

After data preparation file's structure will be as following:
```
+-- data
|   +-- train
|   +-- test
|   train.txt
|   test.txt
```

Folders *train* and *test* contain images and corresponding *.txt* files with the same names as images. In every *.txt* file there are 5 numbers that are number of *class* and *coordinates* of *boundary box*. In files *train.txt* and *test.txt* there are full paths to the images that will be used in *Darknet* framework. After *Darknet* installation we will update configuration and set the paths to these files in order *Darknet* knows with which data to train and test.

<br/>

### <a id="ssd">SSD</a>

<br/>

### <a id="dssd">DSSD</a>

<br/>

### <a id="r-fcn">R-FCN</a>

<br/>

### <a id="r-cnn">R-CNN</a>

<br/>

### <a id="fast-r-cnn">Fast R-CNN</a>

<br/>

### <a id="faster-r-cnn">Faster R-CNN</a>

<br/>

### <a id="mask-r-cnn">Mask R-CNN</a>

<br/>

### <a id="nasnet">NASNet</a>

<br/>

### <a id="fpn-frcn">FPN FRCN</a>

<br/>

### <a id="retinanet">Retinanet</a>

<br/>

### MIT License
### Copyright (c) 2019 Valentyn N Sichkar
### github.com/sichkar-valentyn
### Reference to:
Valentyn N Sichkar. Neural Networks for computer vision in autonomous vehicles and robotics // GitHub platform. DOI: 10.5281/zenodo.1317904
