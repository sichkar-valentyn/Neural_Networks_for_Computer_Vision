# File: CNN_More_complex_example.py
# Description: Neural Networks for computer vision in autonomous vehicles and robotics
# Environment: PyCharm and Anaconda environment
#
# MIT License
# Copyright (c) 2018 Valentyn N Sichkar
# github.com/sichkar-valentyn
#
# Reference to:
# Valentyn N Sichkar. Neural Networks for computer vision in autonomous vehicles and robotics // GitHub platform. DOI: 10.5281/zenodo.1317904




# Implementing more complex example of convolution
# Input image is GrayScale with three identical channels

# Hyperparameters is as following:
# Filter (kernel) size, K_size = 3
# Step for sliding (stride), Step = 1
# Processing edges (zero valued frame around image), Pad = 1
# Consequently, output image size is as following:
# Width_Out = (Width_In - K_size + 2*Pad) / Step + 1
# Height_Out = (Height_In - K_size + 2*Pad) / Step + 1
# If an input image is 50x50 spatial size (width and height), then output image:
# Width_Out = Height_Out = (50 - 3 + 2*1)/1 + 1 = 50
# The shape of output image is the same with input image according to the chosen Hyperparameters


# Importing needed libraries
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


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


# Creating function for ReLU Layer
def relu_layer(maps):
    # Preparing array for output result
    r = np.zeros_like(maps)
    # Using 'np.where' setting condition that every element in 'maps' has to be more than appropriate element in 'r'
    result = np.where(maps > r, maps, r)
    # Returning resulted array
    return result


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


# Opening grayscale input image and putting data into array
input_image = Image.open("images/eagle_grayscale.jpeg")
image_np = np.array(input_image)
# Checking the shape of the array
print(image_np.shape)  # (270, 480, 3)
# Checking if all channels are the same
print(np.array_equal(image_np[:, :, 0], image_np[:, :, 1]))  # True
print(np.array_equal(image_np[:, :, 1], image_np[:, :, 2]))  # True


# Option #1 for filters
# Creating 4 first filters for the first CNN Layer with random integer numbers in range [-1, 1]
# The depth of each filter has to match the number of channels (depth) in input image
# In our case it is 3 as image has three identical grayscale channels
filter_1 = np.random.random_integers(low=-1, high=1, size=(4, 3, 3, image_np.shape[-1]))
# 4 corresponds to number of filters
# 3 and another 3 corresponds to spatial size of filters - width and height
# image_np.shape[-1] corresponds to the depth of each volume of filters
# Checking the shape of the filters
print(filter_1.shape)  # (4, 3, 3, 3)

# Option #2 for filters
# Creating filters manually
# The depth of each filter has to match the number of channels (depth) in input image
# In our case it is 3 as image has three identical grayscale channels
filter_1 = np.zeros((4, 3, 3, 3))
# First filter
filter_1[0, :, :, 0] = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
filter_1[0, :, :, 1] = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
filter_1[0, :, :, 2] = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
# Second filter
filter_1[1, :, :, 0] = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
filter_1[1, :, :, 1] = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
filter_1[1, :, :, 2] = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
# Third filter
filter_1[2, :, :, 0] = np.array([[1, -1, 0], [-1, 0, 1], [-1, 0, 1]])
filter_1[2, :, :, 1] = np.array([[1, -1, 0], [-1, 0, 1], [-1, 0, 1]])
filter_1[2, :, :, 2] = np.array([[1, -1, 0], [-1, 0, 1], [-1, 0, 1]])
# Fourth filter
filter_1[3, :, :, 0] = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
filter_1[3, :, :, 1] = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
filter_1[3, :, :, 2] = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
print(filter_1.shape)  # (4, 3, 3, 3)

# CNN Layer number 1
cnn_1 = cnn_layer(image_np, filter_1, pad=1, step=1)
# Replacing all pixel elements in the resulted volume that are more than 255 with 255
cnn_1 = image_pixels_255(cnn_1)
print(cnn_1.shape)  # (270, 480, 4)

# ReLU Layer number 1
# As input we put resulted feature maps from previous CNN Layer
relu_1 = relu_layer(cnn_1)
print(relu_1.shape)  # (270, 480, 4)

# Pooling Layer number 1
# As input we put resulted feature maps from previous ReLU Layer
pooling_1 = pooling_layer(relu_1, size=2, step=2)
print(pooling_1.shape)  # (135, 240, 4)


# Creating 4 filters for the second CNN Layer with random integer numbers in range [-1, 1]
# The depth of each filter has to match the number of channels (depth) in input image volume
# In our case it is 4 as number of output feature maps from first CNN Layer is 4
filter_2 = np.random.random_integers(low=-1, high=1, size=(4, 3, 3, cnn_1.shape[-1]))
# 4 corresponds to number of filters
# 3 and another 3 corresponds to spatial size of filters - width and height
# image_np.shape[-1] corresponds to the depth of each volume of filters
# Checking the shape of the filters
print(filter_2.shape)  # (4, 3, 3, 4)

# CNN Layer number 2
# As input we put resulted feature maps from previous Pooling Layer
cnn_2 = cnn_layer(pooling_1, filter_2, pad=1, step=1)
# Replacing all pixel elements in the resulted volume that are more than 255 with 255
cnn_2 = image_pixels_255(cnn_2)
print(cnn_2.shape)  # (135, 240, 4)

# ReLU Layer number 2
# As input we put resulted feature maps from previous CNN Layer
relu_2 = relu_layer(cnn_2)
print(relu_2.shape)  # (135, 240, 4)

# Pooling Layer number 2
# As input we put resulted feature maps from previous ReLU Layer
pooling_2 = pooling_layer(relu_2, size=2, step=2)
print(pooling_2.shape)  # (67, 120, 4)


# Creating 4 filters for the third CNN Layer with random integer numbers in range [-1, 1]
# The depth of each filter has to match the number of channels (depth) in input image volume
# In our case it is 4 as number of output feature maps from second CNN Layer is 4
filter_3 = np.random.random_integers(low=-1, high=1, size=(4, 3, 3, cnn_2.shape[-1]))
# 4 corresponds to number of filters
# 3 and another 3 corresponds to spatial size of filters - width and height
# image_np.shape[-1] corresponds to the depth of each volume of filters
# Checking the shape of the filters
print(filter_3.shape)  # (4, 3, 3, 4)

# CNN Layer number 3
# As input we put resulted feature maps from previous Pooling Layer
cnn_3 = cnn_layer(pooling_2, filter_3, pad=1, step=1)
# Replacing all pixel elements in the resulted volume that are more than 255 with 255
cnn_3 = image_pixels_255(cnn_3)
print(cnn_3.shape)  # (67, 120, 4)

# ReLU Layer number 3
# As input we put resulted feature maps from previous CNN Layer
relu_3 = relu_layer(cnn_3)
print(relu_3.shape)  # (67, 120, 4)

# Pooling Layer number 3
# As input we put resulted feature maps from previous ReLU Layer
pooling_3 = pooling_layer(relu_3, size=2, step=2)
print(pooling_3.shape)  # (33, 60, 4)


# Showing results on the appropriate figures
n_rows = cnn_1.shape[-1]
figure_1, ax = plt.subplots(nrows=n_rows, ncols=9, edgecolor='Black')
# ax is as (4, 9) np array and we can call each time ax[0, 0]

# Adjusting subplots for CNN Layer number 1
ax[0, 0].imshow(cnn_1[:, :, 0], cmap=plt.get_cmap('gray'))
ax[0, 0].set_axis_off()
ax[0, 0].set_title('CNN #1')
for i in range(1, n_rows):
    ax[i, 0].imshow(cnn_1[:, :, i], cmap=plt.get_cmap('gray'))
    ax[i, 0].set_axis_off()

# Adjusting subplots for ReLU Layer number 1
ax[0, 1].imshow(relu_1[:, :, 0], cmap=plt.get_cmap('gray'))
ax[0, 1].set_axis_off()
ax[0, 1].set_title('ReLU #1')
for i in range(1, n_rows):
    ax[i, 1].imshow(relu_1[:, :, i], cmap=plt.get_cmap('gray'))
    ax[i, 1].set_axis_off()

# Adjusting subplots for Pooling Layer number 1
ax[0, 2].imshow(pooling_1[:, :, 0], cmap=plt.get_cmap('gray'))
ax[0, 2].set_axis_off()
ax[0, 2].set_title('Pooling #1')
for i in range(1, n_rows):
    ax[i, 2].imshow(pooling_1[:, :, i], cmap=plt.get_cmap('gray'))
    ax[i, 2].set_axis_off()

# Adjusting subplots for CNN Layer number 2
ax[0, 3].imshow(cnn_2[:, :, 0], cmap=plt.get_cmap('gray'))
ax[0, 3].set_axis_off()
ax[0, 3].set_title('CNN #2')
for i in range(1, n_rows):
    ax[i, 3].imshow(cnn_2[:, :, i], cmap=plt.get_cmap('gray'))
    ax[i, 3].set_axis_off()

# Adjusting subplots for ReLU Layer number 2
ax[0, 4].imshow(relu_2[:, :, 0], cmap=plt.get_cmap('gray'))
ax[0, 4].set_axis_off()
ax[0, 4].set_title('ReLU #2')
for i in range(1, n_rows):
    ax[i, 4].imshow(relu_2[:, :, i], cmap=plt.get_cmap('gray'))
    ax[i, 4].set_axis_off()

# Adjusting subplots for Pooling Layer number 2
ax[0, 5].imshow(pooling_2[:, :, 0], cmap=plt.get_cmap('gray'))
ax[0, 5].set_axis_off()
ax[0, 5].set_title('Pooling #2')
for i in range(1, n_rows):
    ax[i, 5].imshow(pooling_2[:, :, i], cmap=plt.get_cmap('gray'))
    ax[i, 5].set_axis_off()

# Adjusting subplots for CNN Layer number 3
ax[0, 6].imshow(cnn_3[:, :, 0], cmap=plt.get_cmap('gray'))
ax[0, 6].set_axis_off()
ax[0, 6].set_title('CNN #3')
for i in range(1, n_rows):
    ax[i, 6].imshow(cnn_3[:, :, i], cmap=plt.get_cmap('gray'))
    ax[i, 6].set_axis_off()

# Adjusting subplots for ReLU Layer number 3
ax[0, 7].imshow(relu_3[:, :, 0], cmap=plt.get_cmap('gray'))
ax[0, 7].set_axis_off()
ax[0, 7].set_title('ReLU #3')
for i in range(1, n_rows):
    ax[i, 7].imshow(relu_3[:, :, i], cmap=plt.get_cmap('gray'))
    ax[i, 7].set_axis_off()

# Adjusting subplots for Pooling Layer number 3
ax[0, 8].imshow(pooling_3[:, :, 0], cmap=plt.get_cmap('gray'))
ax[0, 8].set_axis_off()
ax[0, 8].set_title('Pooling #3')
for i in range(1, n_rows):
    ax[i, 8].imshow(pooling_3[:, :, i], cmap=plt.get_cmap('gray'))
    ax[i, 8].set_axis_off()

# Giving the name to the window with figure
figure_1.canvas.set_window_title('CNN --> ReLU --> Pooling')
# Showing the plots
plt.show()
