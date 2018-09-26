# OpenCV with Python
Implementing OpenCV in Python for Computer Vision tasks. **Coming soon. Release in October 2018**
<br/>[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1317904.svg)](https://doi.org/10.5281/zenodo.1317904)

## Content
Theory and experimental results (on this page):

* <a href="#The easiest ways to install OpenCV for Linux Ubuntu">The easiest ways to install OpenCV for Linux Ubuntu</a>
* <a href="#OpenCV code example">OpenCV code example</a>

<br/>

### <a name="The easiest ways to install OpenCV for Linux Ubuntu">The easiest ways to install OpenCV for Linux Ubuntu</a>
There are few the most simple ways to install OpenCV for Linux, and they are:
* by using `conda`;
* by using `pip3` for python 3;

With first one it's needed to have `conda` been installed and to use one of the following commands:
<br/>`conda install -c conda-forge opencv`
<br/>`conda install -c conda-forge/label/broken opencv`

<br/>With second one use following command:
<br/>`pip3 install opencv-python`

<br/>OpenCV will be installed in choosen environment.
<br/>To check if OpenCV was installed, run python in any form and try this command:
<br/>`import cv2`
<br/>`print(cv2.__version__)`

<br/>

### <a name="OpenCV code example">OpenCV code example</a>

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
