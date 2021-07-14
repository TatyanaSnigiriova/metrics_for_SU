from __future__ import print_function
import cv2 as cv
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, exists

# Read image given by user
images_dir_path = join(".", "test_metrics")
main_image_path = join(images_dir_path, "HR.png")
gen_images_dir_path = join(images_dir_path, "compare")
if not exists(gen_images_dir_path):
    makedirs(gen_images_dir_path)
brightness_images_dir_path = join(gen_images_dir_path, "brightness")
if not exists(brightness_images_dir_path):
    makedirs(brightness_images_dir_path)
contrast_images_dir_path = join(gen_images_dir_path, "contrast")
if not exists(contrast_images_dir_path):
    makedirs(contrast_images_dir_path)
image_format = main_image_path[main_image_path.rfind('.'):]

image = cv.imread(main_image_path)
if image is None:
    print('Could not open or find the image: ', main_image_path)
    exit(0)
new_image = np.zeros(image.shape, image.dtype)

# alpha - Simple contrast control
# beta - Simple brightness control
# Do the operation new_image(i,j) = alpha*image(i,j) + beta
image_plus_1_5_contrast = cv.convertScaleAbs(image, alpha=1.5, beta=0)
image_plus_2_contrast = cv.convertScaleAbs(image, alpha=2., beta=0)
image_plus_2_5_contrast = cv.convertScaleAbs(image, alpha=2.5, beta=0)
image_minus_0_5_contrast = cv.convertScaleAbs(image, alpha=0.5, beta=0)
image_minus_1_contrast = cv.convertScaleAbs(image, alpha=0., beta=0)
image_minus_1_5_contrast = cv.convertScaleAbs(image, alpha=-0.5, beta=0)

image_plus_25_brightness = cv.convertScaleAbs(image, alpha=1., beta=25)
image_plus_50_brightness = cv.convertScaleAbs(image, alpha=1., beta=50)
image_plus_75_brightness = cv.convertScaleAbs(image, alpha=1., beta=75)
image_minus_25_brightness = cv.convertScaleAbs(image, alpha=1., beta=-25)
image_minus_50_brightness = cv.convertScaleAbs(image, alpha=1., beta=-50)
image_minus_75_brightness = cv.convertScaleAbs(image, alpha=1., beta=-75)

cv.imwrite(join(contrast_images_dir_path, "+1_5" + image_format), image_plus_1_5_contrast)
cv.imwrite(join(contrast_images_dir_path, "+2" + image_format), image_plus_2_contrast)
cv.imwrite(join(contrast_images_dir_path, "+2_5" + image_format), image_plus_2_5_contrast)
cv.imwrite(join(contrast_images_dir_path, "-0_5" + image_format), image_minus_0_5_contrast)
cv.imwrite(join(contrast_images_dir_path, "-1" + image_format), image_minus_1_contrast)
cv.imwrite(join(contrast_images_dir_path, "-1_5" + image_format), image_minus_1_5_contrast)

cv.imwrite(join(brightness_images_dir_path, "+25" + image_format), image_plus_25_brightness)
cv.imwrite(join(brightness_images_dir_path, "+50" + image_format), image_plus_50_brightness)
cv.imwrite(join(brightness_images_dir_path, "+75" + image_format), image_plus_75_brightness)
cv.imwrite(join(brightness_images_dir_path, "-25" + image_format), image_minus_25_brightness)
cv.imwrite(join(brightness_images_dir_path, "-50" + image_format), image_minus_50_brightness)
cv.imwrite(join(brightness_images_dir_path, "-75" + image_format), image_minus_75_brightness)
