# File: CNN_Simple_Convolution.py
# Description: Neural Networks for computer vision in autonomous vehicles and robotics
# Environment: PyCharm and Anaconda environment
#
# MIT License
# Copyright (c) 2018 Valentyn N Sichkar
# github.com/sichkar-valentyn
#
# Reference to:
# Valentyn N Sichkar. Neural Networks for computer vision in autonomous vehicles and robotics // GitHub platform. DOI: 10.5281/zenodo.1317904




# Implementing simple convolution to the GreyScaled image

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

# Creating a figure with subplots to show all channels separately
figure_0, ax = plt.subplots(nrows=2, ncols=2)
# ax is (2, 2) np array and to make it easier to read we use 'flatten' function
# Or we can call each time ax[0, 0]
ax0, ax1, ax2, ax3 = ax.flatten()

# Adjusting first subplot
ax0.imshow(channel_0, cmap=plt.get_cmap('gray'))
ax0.set_xlabel('')
ax0.set_ylabel('')
ax0.set_title('First channel')

# Adjusting second subplot
ax1.imshow(channel_1, cmap=plt.get_cmap('gray'))
ax1.set_xlabel('')
ax1.set_ylabel('')
ax1.set_title('Second channel')

# Adjusting third subplot
ax2.imshow(channel_2, cmap=plt.get_cmap('gray'))
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.set_title('Third channel')

# Adjusting fourth subplot
ax3.imshow(image_np, cmap=plt.get_cmap('gray'))
ax3.set_xlabel('')
ax3.set_ylabel('')
ax3.set_title('Original image')

# Function to make distance between figures
plt.tight_layout()
# Giving the name to the window with figure
figure_0.canvas.set_window_title('GreyScaled image with three identical channels')
# Showing the plots
plt.show()


# Preparing image for convolution
# In order to get feature map (convolved output image) in the same size, it is needed to set Hyperparameters
# Filter (kernel) size, K_size = 3
# Step for sliding (stride), Step = 1
# Processing edges (zero valued frame around image), Pad = 1
# Consequently, output image size is (width and height are the same):
# Width_Out = (Width_In - K_size + 2*Pad)/Step + 1
# Imagine, that input image is 5x5 spatial size (width and height), then output image:
# Width_Out = (5 - 3 + 2*1)/1 + 1 = 5, and this is equal to input image

# Taking as input image first channel as array
input_image = image_np[:, :, 0]
# Checking the shape
print(input_image.shape)  # (1280, 830)

# Applying to the input image Pad frame with zero values
# Using NumPy method 'pad'
input_image_with_pad = np.pad(input_image, (1, 1), mode='constant', constant_values=0)
# Checking the shape
print(input_image_with_pad.shape)  # (1282, 832)

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


# Implementing another standard filters for edge detection
# Defining standard filters (kernel) with size 3x3 for edge detection
filter_1 = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
filter_2 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
filter_3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
# Checking the shape
print(filter_1.shape, filter_2.shape, filter_3.shape)  # (3, 3) (3, 3) (3, 3)


# In order to prevent appearing values that are more than 255
# And in general keep values for image pixels in range from 0 to 255
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

# Showing results on the appropriate figures
figure_1, ax = plt.subplots(nrows=2, ncols=2)
# ax is (2, 2) np array and to make it easier to read we use 'flatten' function
# Or we can call each time ax[0, 0]
ax0, ax1, ax2, ax3 = ax.flatten()

# Adjusting first subplot
ax0.imshow(output_image_1, cmap=plt.get_cmap('gray'))
ax0.set_xlabel('')
ax0.set_ylabel('')
ax0.set_title('Filter #1')

# Adjusting second subplot
ax1.imshow(output_image_2, cmap=plt.get_cmap('gray'))
ax1.set_xlabel('')
ax1.set_ylabel('')
ax1.set_title('Filter #2')

# Adjusting third subplot
ax2.imshow(output_image_3, cmap=plt.get_cmap('gray'))
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.set_title('Filter #3')

# Adjusting fourth subplot
ax3.imshow(input_image, cmap=plt.get_cmap('gray'))
ax3.set_xlabel('')
ax3.set_ylabel('')
ax3.set_title('Original image')

# Function to make distance between figures
plt.tight_layout()
# Giving the name to the window with figure
figure_1.canvas.set_window_title('Convolution with filters for edge detection')
# Showing the plots
plt.show()
