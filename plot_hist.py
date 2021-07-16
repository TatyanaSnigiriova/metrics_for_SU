from matplotlib import pyplot as plt
from os.path import join
import cv2 as cv
import numpy as np


def plot_hists(
        minus_shift_image_path, zero_shift_image_path, plus_shift_image_path,
        minus_shift_hist_title, plus_shift_hist_title
):
    image_minus_shift = cv.imread(minus_shift_image_path, cv.IMREAD_GRAYSCALE)
    image_zero_shift = cv.imread(zero_shift_image_path, cv.IMREAD_GRAYSCALE)
    image_plus_shift = cv.imread(plus_shift_image_path, cv.IMREAD_GRAYSCALE)

    fig_shift, axes_shift = plt.subplots(1, 3, figsize=(25, 25), sharey=True)
    counts_minus_shift, bins_minus_shift = np.histogram(image_minus_shift, bins=range(0, 256, 1))
    print(counts_minus_shift[0])
    counts_minus_shift[0] = 0
    counts_zero_shift, bins_zero_shift = np.histogram(image_zero_shift, bins=range(0, 256, 1))
    counts_plus_shift, bins_plus_shift = np.histogram(image_plus_shift, bins=range(0, 256, 1))
    print(counts_plus_shift[-1])
    counts_plus_shift[-1] = 0

    axes_shift[0].hist(bins_minus_shift[:-1], bins_minus_shift, weights=counts_minus_shift)
    axes_shift[0].plot((128, 128), (0, max(counts_minus_shift)), linestyle="--")
    axes_shift[0].set_title(minus_shift_hist_title)
    axes_shift[1].hist(bins_zero_shift[:-1], bins_zero_shift, weights=counts_zero_shift)
    axes_shift[1].set_title('No preprocessing image histogram')
    axes_shift[1].plot((128, 128), (0, max(counts_minus_shift)), linestyle="--")
    axes_shift[2].hist(bins_plus_shift[:-1], bins_plus_shift, weights=counts_plus_shift)
    axes_shift[2].set_title(plus_shift_hist_title)
    axes_shift[2].plot((128, 128), (0, max(counts_minus_shift)), linestyle="--")

    plt.show()


def main():
    main_dir = join(".", "test_metrics", "compare_M2_1")
    brightness_path = join(main_dir, "brightness")
    plot_hists(
        join(brightness_path, "-75.png"),
        join(brightness_path, "0.png"),
        join(brightness_path, "75.png"),
        'Image + bias=-75 histogram',
        'Image + bias=75 histogram'
    )

    contrast_path = join(main_dir, "contrast")
    plot_hists(
        join(contrast_path, "0.6.png"),
        join(contrast_path, "1.0.png"),
        join(contrast_path, "1.4.png"),
        'Image * alpha=0.6 histogram',
        'Image * alpha=1.4 histogram'
    )

    gamma_contrast_path = join(main_dir, "gamma_contrast")
    plot_hists(
        join(gamma_contrast_path, "0.6.png"),
        join(gamma_contrast_path, "1.0.png"),
        join(gamma_contrast_path, "1.4.png"),
        'Histogram for gamma=0.6',
        'Histogram for gamma=1.4'
    )


if __name__ == '__main__':
    # Code to be run only when run directly
    main()
