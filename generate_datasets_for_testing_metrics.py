from __future__ import print_function
import cv2 as cv
import numpy as np
from os import makedirs
from os.path import join, exists

# Source: https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
# Read image given by user
images_dir_path = join(".", "test_metrics")
main_image_path = join(images_dir_path, "HR.png")
image_format = main_image_path[main_image_path.rfind('.'):]
main_image = cv.imread(main_image_path)
if main_image is None:
    print('Could not open or find the image: ', main_image_path)
    exit(0)

gen_images_dir_path = join(images_dir_path, "compare_2")
if not exists(gen_images_dir_path):
    makedirs(gen_images_dir_path)

# alpha - Simple contrast control
# beta - Simple brightness control
# Do the operation new_image(i,j) = alpha*image(i,j) + beta
contrast_images_dir_path = join(gen_images_dir_path, "contrast")
if not exists(contrast_images_dir_path):
    makedirs(contrast_images_dir_path)
for alpha in np.arange(0., 5.1, 0.2):
    alpha = round(alpha, 1)
    image_shift_contrast = cv.convertScaleAbs(main_image, alpha=alpha, beta=0)
    cv.imwrite(join(contrast_images_dir_path, str(alpha) + image_format), image_shift_contrast)

brightness_images_dir_path = join(gen_images_dir_path, "brightness")
if not exists(brightness_images_dir_path):
    makedirs(brightness_images_dir_path)
for beta in range(-510, 511, 15):
    image_shift_brightness = cv.convertScaleAbs(main_image, alpha=1., beta=beta)
    if any(image_shift_brightness[image_shift_brightness != 255]):
        cv.imwrite(join(brightness_images_dir_path, str(beta) + image_format), image_shift_brightness)

gamma_contrast_images_dir_path = join(gen_images_dir_path, "gamma_contrast")
if not exists(gamma_contrast_images_dir_path):
    makedirs(gamma_contrast_images_dir_path)
for gamma in np.arange(0.1, 2.1, 0.1):
    gamma = round(gamma, 1)
    image_shift_gamma = np.array(255 * (main_image / 255) ** gamma, dtype='uint8')
    cv.imwrite(join(gamma_contrast_images_dir_path, str(gamma) + image_format), image_shift_gamma)
