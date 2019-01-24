# OpenCV with Python
Implementing OpenCV in Python for Computer Vision tasks.
<br/>[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1317904.svg)](https://doi.org/10.5281/zenodo.1317904)

## Content
Theory and experimental results (on this page):

* [The easiest ways to install OpenCV for Linux Ubuntu](#the-easiest-ways-to-install-opencv-for-linux-ubuntu)
* [Brightness and Rotation changes](#brightness-and-rotation-changes)

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

### <a id="brightness-and-rotation-changes">Brightness and Rotation changes</a>
Easy way to rotate image and change brightness with **OpenCV**.
<br/>Can be used for preprocessing datasets for **Classification Tasks**.

```py
# Importing needed libraries
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2 


# Reading image with PIL library
image_PIL = Image.open('images/stop_sign.jpg')
# Converting read image into numpy array
image_PIL_np = np.array(image_PIL)

# Reading image with OpenCV library
# With OpenCV image is opened already as numpy array
# WARNING! OpenCV by default reads images in BGR format
image_OpenCV = cv2.imread('images/stop_sign.jpg', 1)
# Converting BGR to RGB
image_OpenCV = cv2.cvtColor(image_OpenCV, cv2.COLOR_BGR2RGB)

# Checking if arrays are equal
print(np.array_equal(image_OpenCV, image_OpenCV))  # True

# Showing image in OpenCV window
cv2.namedWindow('Traffic Sign', cv2.WINDOW_NORMAL)  # Specifing that window is resizable
# Showing image
# WARNING! 'cv2.imshow' takes images in BGR format
# Consequently, we need to convert image from RGB to BGR
cv2.imshow('Traffic Sign', cv2.cvtColor(image_OpenCV, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)  # Waiting for any key being pressed
cv2.destroyWindow('Traffic Sign')
# cv2.destroyAllWindows()


# Defining function for changing brightness
def brightness_changement(image):
    # Converting firstly image from RGB to HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Defining random value for changing brightness
    random_brightness = np.random.uniform()
    # Implementing changing of Value channel of HSV image
    image_hsv[:, :, 2] = image_hsv[: , :, 2] * random_brightness
    # Converting HSV changed image to RGB
    image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    # Returning image with changed brightness
    return image_rgb


"""
To rotate an image using OpenCV Python,
first, calculate the affine matrix that does the affine transformation (linear mapping of pixels),
then warp the input image with the affine matrix.

Example:

M = cv2.getRotationMatrix2D(center, angle, scale)
rotated = cv2.warpAffine(img, M, (w, h))

where

center:  center of the image (the point about which rotation has to happen)
angle:   angle by which image has to be rotated in the anti-clockwise direction
scale:   1.0 mean, the shape is preserved. Other value scales the image by the value provided
rotated: ndarray that holds the rotated image data
"""

# Defining function for changing rotation of image
def rotation_changement(image, angle):
    # Getting shape of image
    rows, columns, channels = image.shape    
    # Implementing rotation
    # Calculating Affine Matrix
    affine_matrix = cv2.getRotationMatrix2D((columns / 2, rows / 2), angle, 1)
    # Warping original image with Affine Matrix
    rotated_image = cv2.warpAffine(image, affine_matrix, (columns, rows))
    # Returning rotated image
    return rotated_image


# Plotting RGB example with changes
# Pay attention!
# In OpenCV conventional ranges for R, G, and B channel values are:
#     0 to 255 for CV_8U images
#     0 to 65535 for CV_16U images
#     0 to 1 for CV_32F images
# Therefore if input image has values in [0..255] range,
# It is needed to convert it to uint8 dtype explicitly:
# im = np.random.randint(0,255,(224,224,3)).astype(np.uint8)


figure, ax = plt.subplots(nrows=2, ncols=4)
# 'ax 'is as (2, 4) np.array and we can call each time ax[0, 0]
# Plotting image with rotations
for i in range(4):
    ax[0, i].set_axis_off()
    ax[0, i].set_title(str(90 * (i +1)))
    ax[0, i].imshow(rotation_changement(image_OpenCV, 90 * (i + 1)))

# Plotting image with changing brightness
for i in range(4):
    ax[1, i].set_axis_off()
    ax[1, i].imshow(brightness_changement(image_OpenCV))

plt.show()
```

![brightness_rotation_changing.png](https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision/blob/master/images/brightness_rotation_changing.png)

<br/>

### MIT License
### Copyright (c) 2019 Valentyn N Sichkar
### github.com/sichkar-valentyn
### Reference to:
Valentyn N Sichkar. Neural Networks for computer vision in autonomous vehicles and robotics // GitHub platform. DOI: 10.5281/zenodo.1317904
