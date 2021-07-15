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
for alpha in np.arange(0., 5.1, 0.2):
    alpha = round(alpha, 2)
    image_shift_contrast = cv.convertScaleAbs(image, alpha=alpha, beta=0)
    cv.imwrite(join(contrast_images_dir_path, str(alpha) + image_format), image_shift_contrast)

for beta in np.arange(-510, 511, 15):
    image_shift_brightness = cv.convertScaleAbs(image, alpha=1., beta=beta)
    if any(image_shift_brightness[image_shift_brightness != 255]):
        cv.imwrite(join(brightness_images_dir_path, str(beta) + image_format), image_shift_brightness)
