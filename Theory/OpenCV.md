# OpenCV with Python
Implementing OpenCV in Python for Computer Vision tasks. **Coming soon. Will be released in January 2019.**
<br/>[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1317904.svg)](https://doi.org/10.5281/zenodo.1317904)

## Content
Theory and experimental results (on this page):

* [The easiest ways to install OpenCV for Linux Ubuntu](#the-easiest-ways-to-install-opencv-for-linux-ubuntu)
* [OpenCV code example](#opencv-code-example)

<br/>

### <a id="the-easiest-ways-to-install-opencv-for-linux-ubuntu">The easiest ways to install OpenCV for Linux Ubuntu</a>
There are few the most simple ways to install OpenCV for Linux Ubuntu, and they are:
* by using `conda`
* by using `pip3` for python 3

With first one it's needed to have `conda` been installed and to use one of the following commands:
* `conda install -c conda-forge opencv`
* `conda install -c conda-forge/label/broken opencv`

With second one use one of the following commands:
* `pip3 install opencv-python` - only main modules
* `pip install opencv-contrib-python` - both main and contrib modules

Also, might need to be installed following, if there is mistake arise:
* `sudo apt-get install libsm6`
* `sudo apt-get install -y libxrender-dev`

OpenCV will be installed in choosen environment.
<br/>To check if OpenCV was installed, run python in any form and run following two lines of code:
```py
import cv2
print(cv2.__version__)
```

As a result the version of installed OpenCV has to be shown, like this:
<br/>`3.4.3`

<br/>

### <a id="opencv-code-example">OpenCV code example</a>

```py
import numpy as np

```

Full code is available here: 

<br/>

### MIT License
### Copyright (c) 2019 Valentyn N Sichkar
### github.com/sichkar-valentyn
### Reference to:
Valentyn N Sichkar. Neural Networks for computer vision in autonomous vehicles and robotics // GitHub platform. DOI: 10.5281/zenodo.1317904
