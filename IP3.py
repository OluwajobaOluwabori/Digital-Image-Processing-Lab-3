import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import tight_layout
from scipy import ndimage
from skimage.exposure import rescale_intensity, equalize_hist
from skimage.util import img_as_float

# Read the image
bacteria_image = cv2.imread('bacteria.jpg')
bacteria_array = np.array(bacteria_image)

# print("Standard Deviation:", ndimage.standard_deviation(bacteria_array))
# print("Mean:", ndimage.mean(bacteria_array))
# print("Median:", ndimage.median(bacteria_array))

# 1.2
chro_image = cv2.imread('chro.bmp', cv2.IMREAD_GRAYSCALE)

chro_image_flatten = chro_image.flatten() # Flattens the image for histogram calculation
# Calculate the histogram and cumulative histogram
hist, bin_edges = np.histogram(chro_image)
cumulative_hist = np.cumsum(hist)

total_pixels = cumulative_hist[-1] # Total number of pixels
# Find the median value
median_index = np.searchsorted(cumulative_hist, total_pixels / 2)
median_value = bin_edges[median_index]
print("Median Value from Cumulative Histogram:", median_value)

fig, axs = plt.subplots(3, 4, tight_layout=True)
axs[0,0].imshow(chro_image, cmap='gray')
axs[0,0].set_title('Original Image')
axs[0,1].hist(chro_image_flatten, bins=64)
axs[0,1].set_title('Original Image Histogram with 64 Bins')
axs[0,2].hist(chro_image_flatten, bins=16)
axs[0,2].set_title('Original Image Histogram with 16 Bins')
axs[0,3].hist(chro_image_flatten, bins=16, cumulative = True)
axs[0,3].set_title('Original Image Cumulative Histogram- 16 bins')

Rotated_image = cv2.rotate(chro_image, cv2.ROTATE_90_CLOCKWISE)

axs[1,0].imshow(Rotated_image, cmap='gray')
axs[1,0].set_title('Rotated Image')
axs[1,1].hist(Rotated_image.flatten(), bins=64)
axs[1,1].set_title('Rotated Image Histogram with 64 Bins')
axs[1,2].hist(Rotated_image.flatten(), bins=16)
axs[1,2].set_title('Rotated Image Histogram with 16 Bins')
axs[1,3].hist(Rotated_image.flatten(), cumulative = True)
axs[1,3].set_title('Rotated Image Cumulative Histogram- 16 bins')

inverted_image = cv2.bitwise_not(chro_image)

axs[2,0].imshow(inverted_image, cmap='gray')
axs[2,0].set_title('Inverted Image')
axs[2,1].hist(inverted_image.flatten(), bins=64)
axs[2,1].set_title('Inverted Image Histogram with 64 Bins')
axs[2,2].hist(inverted_image.flatten(), bins=16)
axs[2,2].set_title('Inverted Image Histogram with 16 Bins')
axs[2,3].hist(inverted_image.flatten(), bins=16, cumulative = True)
axs[2,3].set_title('Inverted Image Cumulative Histogram- 16 bins')

fig.suptitle('Original, Rotated and Inverted Images Histogram')
plt.show()

1.3
a = chro_image.min()
b = chro_image.max()
print("Min: {}, Max: {}".format(a, b))
image_stretched= (chro_image-a)/(b-a)


# Using skimage
stretched_image_sk = rescale_intensity(chro_image, in_range='image')#, out_range=(0, 1)
print("Min: {}, Max: {}".format(stretched_image_sk.min(), stretched_image_sk.max()))

# Plotting the histogram
fig,axs = plt.subplots(2,2)
axs[0,0].imshow(image_stretched,cmap='gray')
axs[0,0].set_title("Stretched Image (custom algorithm)")
axs[0,1].hist(image_stretched.flatten())
axs[0,1].set_xlabel('Intensity value')
axs[0,1].set_ylabel('Number of pixels')
axs[0,1].set_title('Histogram of the custom stretched image')
axs[1,0].imshow(stretched_image_sk, cmap='gray')
axs[1,1].hist(stretched_image_sk.flatten())
axs[1,0].set_title("Stretched Image (skimage)")
axs[1,1].set_title('Histogram of the custom stretched image without specifying output interval')
axs[1,1].set_xlabel('Intensity value')
axs[1,1].set_ylabel('Number of pixels')

fig.tight_layout()
plt.show()

from skimage import io, img_as_float

# 1.4
elaine_image = img_as_float(cv2.imread('elaine.jpg'))
elaineDark_image = img_as_float(cv2.imread('elaineDark.jpg'))
elaineLight_image = img_as_float(cv2.imread('elaineLight.jpg'))
elaineContrasted_image = img_as_float(cv2.imread('elaineContrasted.jpg'))
elaineLowContrasted_image = img_as_float(cv2.imread('elaineLowContrasted.jpg'))

neck_image = img_as_float(cv2.imread('neck.bmp'))
covering_image = img_as_float(cv2.imread('covering.bmp'))


# Equalization
img_eq = equalize_hist(elaine_image)
img_eq_dark = equalize_hist(elaineDark_image)
img_eq_light = equalize_hist(elaineLight_image)
img_eq_contrasted = equalize_hist(elaineContrasted_image)
img_eq_lowcontrasted = equalize_hist(elaineLowContrasted_image)



fig, axs = plt.subplots(5, 4, tight_layout=True)
axs[0, 0].imshow(elaine_image, cmap='gray')
axs[0,0].axis('off')
axs[0, 0].set_title('elaine Original Image')
axs[0, 1].imshow(img_eq, cmap='gray')
axs[0,1].axis('off')
axs[0, 1].set_title('Equalized Image')
axs[0,2].hist(elaine_image.flatten())
axs[0,2].set_title('Original Image Histogram')
axs[0,3].hist(img_eq.flatten())
axs[0,3].set_title('Equalized Image Histogram')
axs[1, 0].imshow(elaineDark_image, cmap='gray')
axs[1,0].axis('off')
axs[1, 0].set_title('elaineDark Original Image')
axs[1, 1].imshow(img_eq_dark, cmap='gray')
axs[1, 1].axis('off')
axs[1, 1].set_title('Equalized Image')
axs[1, 2].hist(elaineDark_image.flatten())
axs[1, 2].set_title('Original Image Histogram')
axs[1, 3].hist(img_eq_dark.flatten())
axs[1, 3].set_title('Equalized Image Histogram')
axs[2, 0].imshow(elaineLight_image, cmap='gray')
axs[2, 0].set_title('elaineLight Original Image')
axs[2, 1].imshow(img_eq_light, cmap='gray')
axs[2, 1].axis('off')
axs[2, 1].set_title('Equalized Image')
axs[2, 2].hist(elaineLight_image.flatten())
axs[2, 2].set_title('Original Image Histogram')
axs[2, 3].hist(img_eq_light.flatten())
axs[2, 3].set_title('Equalized Image Histogram')
axs[3, 0].imshow(elaineContrasted_image, cmap='gray')
axs[3, 0].axis('off')
axs[3, 0].set_title('elaineContrasted Original Image')
axs[3, 1].imshow(img_eq_contrasted, cmap='gray')
axs[3, 1].set_title('Equalized Image')
axs[3, 2].hist(elaineContrasted_image.flatten())
axs[3, 2].axis('off')
axs[3, 2].set_title('Original Image Histogram')
axs[3, 3].hist(img_eq_contrasted.flatten())
axs[3, 3].set_title('Equalized Image Histogram')
axs[4, 0].imshow(elaineLowContrasted_image, cmap='gray')
axs[4, 0].set_title('elaineLowContrasted Original Image')
axs[4, 1].imshow(img_eq_lowcontrasted, cmap='gray')
axs[4, 1].set_title('Equalized Image')
axs[4,2].hist(elaineLowContrasted_image.flatten())
axs[4,2].set_title('Original Image Histogram')
axs[4,3].hist(img_eq_lowcontrasted.flatten())
axs[4, 3].set_title('Equalized Image Histogram')
fig.suptitle('Equalization of Images')
plt.show()
#
# # image = np.array(image)
#
print("Standard Deviation and Mean of elaine:", ndimage.standard_deviation(elaine_image), ndimage.mean(elaine_image))
print("Standard Deviation and Mean of elaine_eq:", ndimage.standard_deviation(img_eq), ndimage.mean(img_eq))
print("Standard Deviation and Mean of elaineDark:", ndimage.standard_deviation(elaineDark_image), ndimage.mean(elaineDark_image))
print("Standard Deviation and Mean of elaineDark_eq:", ndimage.standard_deviation(img_eq_dark), ndimage.mean(img_eq_dark))
print("Standard Deviation and Mean of elaineLight:", ndimage.standard_deviation(elaineLight_image), ndimage.mean(elaineLight_image))
print("Standard Deviation and Mean of elaineLight_eq:", ndimage.standard_deviation(img_eq_light), ndimage.mean(img_eq_light))
print("Standard Deviation and Mean of elaineContrasted:", ndimage.standard_deviation(elaineContrasted_image), ndimage.mean(elaineContrasted_image))
print("Standard Deviation and Mean of elaineContrasted_eq:", ndimage.standard_deviation(img_eq_contrasted), ndimage.mean(img_eq_contrasted))
print("Standard Deviation and Mean of elaineLowContrasted:", ndimage.standard_deviation(elaineLowContrasted_image), ndimage.mean(elaineLowContrasted_image))
print("Standard Deviation and Mean of elaineLowContrasted_eq:", ndimage.standard_deviation(img_eq_lowcontrasted), ndimage.mean(img_eq_lowcontrasted))

neck_eq = equalize_hist(neck_image)
covering_eq = equalize_hist(covering_image)
#
#
print("Standard Deviation and Mean of neck_image:", ndimage.standard_deviation(neck_image), ndimage.mean(neck_image))
print("Standard Deviation and Mean of neck_eq:", ndimage.standard_deviation(neck_eq), ndimage.mean(neck_eq))
print("Standard Deviation and Mean of covering_image:", ndimage.standard_deviation(covering_image), ndimage.mean(covering_image))
print("Standard Deviation and Mean of covering_eq:", ndimage.standard_deviation(covering_eq), ndimage.mean(covering_eq))

# Load color image
from skimage import io, img_as_float
color_image = img_as_float(io.imread('rocks.jpg'))
umbrella_image = img_as_float(io.imread('umbrella.jpg'))


# Equalize each channel
r_eq = equalize_hist(color_image[:, :, 0])
g_eq = equalize_hist(color_image[:, :, 1])
b_eq = equalize_hist(color_image[:, :, 2])

# Combine channels
equalized_color_image = np.stack((r_eq, g_eq, b_eq), axis=2)
print("Standard Deviation and Mean of rocks_image:", ndimage.standard_deviation(color_image), ndimage.mean(color_image))
print("Standard Deviation and Mean of rocks_eq:", ndimage.standard_deviation(equalized_color_image), ndimage.mean(equalized_color_image))

# Equalize each channel
r_eq = equalize_hist(umbrella_image[:, :, 0])
g_eq = equalize_hist(umbrella_image[:, :, 1])
b_eq = equalize_hist(umbrella_image[:, :, 2])

# Combine channels
equalized_umbrella_image = np.stack((r_eq, g_eq, b_eq), axis=2)
print("Standard Deviation and Mean of umbrella_image:", ndimage.standard_deviation(umbrella_image), ndimage.mean(umbrella_image))
print("Standard Deviation and Mean of umbrella_eq:", ndimage.standard_deviation(equalized_umbrella_image), ndimage.mean(equalized_umbrella_image))


fig, axs = plt.subplots(4, 2, tight_layout=True)
axs[0, 0].imshow(neck_image)
axs[0,0].axis('off')
axs[0,0].set_title("Original Neck Image")
axs[0,1].imshow(neck_eq)
axs[0,1].set_title("Equalized Neck Image")
axs[0,1].axis('off')
axs[1,0].imshow(covering_image)
axs[1,0].set_title("Original Covering Image")
axs[1,0].axis('off')
axs[1,1].imshow(covering_eq)
axs[1,1].set_title("Equalized Covering Image")
axs[1,1].axis('off')
axs[2,0].imshow(color_image)
axs[2,0].axis('off')
axs[2,0].set_title("Original Rocks Image")
axs[2,1].imshow(equalized_color_image)
axs[2,1].set_title("Equalized Rocks Image")
axs[2,1].axis('off')
axs[3,0].imshow(umbrella_image)
axs[3,0].axis('off')
axs[3,0].set_title("Original Umbrella Image")
axs[3,1].imshow(equalized_umbrella_image)
axs[3,1].set_title("Equalized Umbrella Image")
axs[3,1].axis('off')
plt.show()



# 2.1

# Read images
gdr_image = cv2.imread('gdr.bmp', cv2.IMREAD_GRAYSCALE)
objects_image = cv2.imread('objects.bmp', cv2.IMREAD_GRAYSCALE)

# Apply manual thresholds
gdr_thresh = 127
objects_thresh = 127
gdr_thresh_isolate = 99
objects_thresh_isolate = -1

gdr_bw = cv2.threshold(gdr_image, gdr_thresh, 255, cv2.THRESH_BINARY)[1]
objects_bw = cv2.threshold(objects_image, objects_thresh, 255, cv2.THRESH_BINARY)[1]
gdr_bw_isolate = cv2.threshold(gdr_image, gdr_thresh_isolate, 255, cv2.THRESH_BINARY)[1]
objects_bw_isolate = cv2.threshold(objects_image, objects_thresh_isolate, 255, cv2.THRESH_BINARY)[1]

# Display results
fig, axs = plt.subplots(2, 3, figsize=(10, 6), tight_layout = True)

axs[0, 0].imshow(gdr_image, cmap='gray')
axs[0, 0].set_title("Original GDR Image")
axs[0, 1].imshow(gdr_bw, cmap='gray')
axs[0, 1].set_title("Thresholded Gdr Image")
axs[0, 2].imshow(gdr_bw_isolate, cmap='gray', vmin=0, vmax=255)
axs[0, 2].set_title("Thresholded Gdr Image (Isolated)")

axs[1, 0].imshow(objects_image, cmap='gray')
axs[1, 0].set_title("Original Objects Image")
axs[1, 1].imshow(objects_bw, cmap='gray')
axs[1, 1].set_title("Thresholded Objects Image")
axs[1, 2].imshow(objects_bw_isolate, cmap='gray', vmin=0, vmax=255)
axs[1, 2].set_title("Thresholded Objects Image (Isolated)")

plt.tight_layout()
plt.show()

2.2


List of image filenames
image_files = ['gdr.bmp', 'objects.bmp', 'bacteria.jpg', 'cell.bmp', 'chro.bmp',
               'gear-wheel.png', 'fibers.jpg', 'fibers2.jpg', 'I10.bmp', 'I12.bmp']

# Process each image
fig, axs = plt.subplots(10, 3, figsize=(15, 10), tight_layout=True)

for i, image_file in enumerate(image_files):    # Load image in grayscale
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    # Apply Otsu's thresholding
    thresh, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Display original image, binary image and histogram
    axs[i, 0].imshow(image, cmap='gray')
    axs[i, 0].axis('off')
    axs[i, 0].set_title(f"Original Image: {image_file.split('.')[0]}",fontsize=10)
    axs[i, 1].imshow(binary_image, cmap='gray')
    axs[i, 1].set_title("Binary Image: Otsu",fontsize=10)
    axs[i, 1].axis('off')
    axs[i, 2].hist(image.flatten(),range=(0,256),bins=256)
    axs[i, 2].set_title(f"Histogram: {image_file.split('.')[0]}",fontsize=10)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, measure, morphology

pills2 = cv2.imread('pills2.bmp', cv2.IMREAD_UNCHANGED)
hsv_pills2 = cv2.cvtColor(pills2, cv2.COLOR_BGR2HSV)

pills2_b = pills2[:, :, 0]
pills2_g = pills2[:, :, 1]
pills2_r = pills2[:, :, 2]

def measure_green_pills_surface_area(image_path):
    # Load the image
    image = io.imread(image_path)

    # Convert the image from RGB to HSV
    hsv_image = color.rgb2hsv(image)

    # Define the range for green color in HSV
    lower_green = np.array([0.25, 0.4, 0.4])  # Adjust these values as needed
    upper_green = np.array([0.7, 1.0, 1.0])  # Adjust these values as needed

    # Create a binary mask where green colors are white and the rest are black
    green_mask = (hsv_image[:, :, 0] >= lower_green[0]) & (hsv_image[:, :, 0] <= upper_green[0]) & \
                 (hsv_image[:, :, 1] >= lower_green[1]) & (hsv_image[:, :, 1] <= upper_green[1]) & \
                 (hsv_image[:, :, 2] >= lower_green[2]) & (hsv_image[:, :, 2] <= upper_green[2])

    # Apply morphological operations to clean up the mask
    green_mask = morphology.remove_small_objects(green_mask, min_size=80)
    green_mask = morphology.remove_small_holes(green_mask, area_threshold=100)

    # Label connected regions in the mask
    labeled_mask, num_labels = measure.label(green_mask, return_num=True, connectivity=2)

    # Measure the area of each labeled region
    regions = measure.regionprops(labeled_mask)
    areas = [region.area for region in regions]

    # Plot the original image and the mask
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(hsv_image)
    plt.title('HSV Image')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(green_mask, cmap='gray')
    plt.title('Green Pills Mask')
    plt.axis('off')
    plt.show()

    return areas
image_path = 'pills2.bmp'
green_pill_areas = measure_green_pills_surface_area(image_path)
print("Surface areas of green pills in pixels:", green_pill_areas)



from skimage import measure, color

# Read the image
image = cv2.imread('chro.bmp', cv2.IMREAD_GRAYSCALE)

# Apply thresholding to create a binary image
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# # Perform connected component analysis
labeled_image, num_objects = measure.label(binary_image, connectivity=2, return_num=True)

# Visualize the labeled image using imshow
plt.figure(figsize=(10, 6))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.title(f'Labeled Image (imshow), Objects: {num_objects}')
plt.imshow(labeled_image, cmap='jet')
plt.axis('off')

# Visualize with label2rgb for better distinction
colored_labels = color.label2rgb(labeled_image, bg_label=0, bg_color=(0, 0, 0))

plt.subplot(1, 3, 3)
plt.imshow(colored_labels, cmap='jet')
plt.title(f'Labeled Image (label2rgb), Objects: {num_objects}')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Number of objects detected: {num_objects}")


# Import necessary libraries
# import cv2
# import matplotlib.pyplot as plt
# from skimage import measure, color

# Read the two images as grayscale
image1 = cv2.imread('original.bmp', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('original2.bmp', cv2.IMREAD_GRAYSCALE)

# Compute the absolute difference between the images
difference = cv2.absdiff(image1, image2)

# Apply thresholding to create a binary mask for differences
_, binary_diff1 = cv2.threshold(difference, 50, 255, cv2.THRESH_BINARY)
_, binary_diff2 = cv2.threshold(difference, 127, 255, cv2.THRESH_BINARY)

# Perform connected component analysis to count distinct regions
labeled_diff_50, num_differences_50 = measure.label(binary_diff1, connectivity=2, return_num=True)
labeled_diff_127, num_differences_127 = measure.label(binary_diff2, connectivity=2, return_num=True)


# Visualize the original images, binary difference, and labeled differences
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(image1, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(image2, cmap='gray')
plt.title("Original2 Image")
plt.axis('off')

plt.subplot(2, 2, 3)
colored_labels_50 = color.label2rgb(labeled_diff_50, bg_label=0, bg_color=(0, 0, 0))
plt.imshow(colored_labels_50)
plt.title(f"Labeled Differences: {num_differences_50}\nThreshold: 50")
plt.axis('off')

plt.subplot(2, 2, 4)
colored_labels_127 = color.label2rgb(labeled_diff_127, bg_label=0, bg_color=(0, 0, 0))
plt.imshow(colored_labels_127)
plt.title(f"Labeled Differences: {num_differences_127}\nThreshold: 127")
plt.axis('off')

plt.tight_layout()
plt.show()

# Print the number of differences
