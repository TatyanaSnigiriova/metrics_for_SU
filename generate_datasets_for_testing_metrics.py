from os import makedirs
from os.path import exists, join
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np


def plot_hists_for_three_shifts(
        minus_shift_image_path, zero_shift_image_path, plus_shift_image_path,
        minus_shift_hist_title, zero_shift_hist_title, plus_shift_hist_title,
        save,
        set_extreme_values_to_zero=True
):
    minus_shift_image = cv.imread(minus_shift_image_path, cv.IMREAD_GRAYSCALE)
    zero_shift_image = cv.imread(zero_shift_image_path, cv.IMREAD_GRAYSCALE)
    plus_shift_image = cv.imread(plus_shift_image_path, cv.IMREAD_GRAYSCALE)
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), sharey=True)
    minus_shift_counts, minus_shift_bins = np.histogram(minus_shift_image, bins=range(0, 256, 1))
    if set_extreme_values_to_zero:
        print(f"hist(minus_shift_image) for zero intensity = {minus_shift_counts[0]}")
        minus_shift_counts[0] = 0

    zero_shift_counts, zero_shift_bins = np.histogram(zero_shift_image, bins=range(0, 256, 1))

    plus_shift_counts, plus_shift_bins = np.histogram(plus_shift_image, bins=range(0, 256, 1))
    if set_extreme_values_to_zero:
        print(f"hist(plus_shift_image) for max intensity = {plus_shift_counts[-1]}")
        plus_shift_counts[-1] = 0
    max_count = max(max(minus_shift_counts), max(zero_shift_counts), max(plus_shift_counts))

    axes[0].hist(minus_shift_bins[:-1], minus_shift_bins, weights=minus_shift_counts)
    axes[0].plot((128, 128), (0, max_count), linestyle="--")
    axes[0].set_title(minus_shift_hist_title)

    axes[1].hist(zero_shift_bins[:-1], zero_shift_bins, weights=zero_shift_counts)
    axes[1].plot((128, 128), (0, max_count), linestyle="--")
    axes[1].set_title(zero_shift_hist_title)


    axes[2].hist(plus_shift_bins[:-1], plus_shift_bins, weights=plus_shift_counts)
    axes[2].plot((128, 128), (0, max_count), linestyle="--")
    axes[2].set_title(plus_shift_hist_title)

    plt.savefig(save)


# Source: https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
def generate_datasets_linearly(
        images_dir_path,
        dataset_name, images_format,
        extended_dataset=True,
        alpha_contrast=True,
        beta_brightness=True,
        plot_hists=True
):
    if plot_hists:
        hists_dir_path = join(images_dir_path, "hists")
        if not exists(hists_dir_path):
            makedirs(hists_dir_path)

    if extended_dataset:
        dir_postfix = 2
    else:
        dir_postfix = 1
    pattern = f"{dataset_name}_{dir_postfix}"

    # The image for which the test datasets were generated
    main_image_name = f"{dataset_name}.{images_format}"
    main_image_path = join(images_dir_path, main_image_name)

    main_image = cv.imread(main_image_path)
    if main_image is None:
        print(
            f"Could not open or find the main image {main_image_path} for which the test datasets should be generated!"
        )
        exit(0)

    gen_images_dir_path = join(images_dir_path, f"compare_{pattern}")
    if not exists(gen_images_dir_path):
        makedirs(gen_images_dir_path)

    # alpha - Simple contrast control
    # beta - Simple brightness control
    # Do the operation new_image(i,j) = alpha*image(i,j) + beta
    if alpha_contrast:
        alpha_contrast_images_dir_path = join(gen_images_dir_path, "alpha_contrast")
        if not exists(alpha_contrast_images_dir_path):
            makedirs(alpha_contrast_images_dir_path)

        if extended_dataset:
            alpha_iterator = np.arange(0., 5.1, 0.2)
        else:
            alpha_iterator = np.arange(0., 2.1, 0.2)

        for alpha in alpha_iterator:
            alpha = round(alpha, 1)
            alpha_contrast_shift_image = cv.convertScaleAbs(main_image, alpha=alpha, beta=0)
            cv.imwrite(join(alpha_contrast_images_dir_path, str(alpha) + "." + images_format),
                       alpha_contrast_shift_image)

        if plot_hists:
            alpha_contrast_hists_dir_path = join(hists_dir_path, "alpha_contrast")
            if not exists(alpha_contrast_hists_dir_path):
                makedirs(alpha_contrast_hists_dir_path)

            plot_hists_for_three_shifts(
                join(alpha_contrast_images_dir_path, f"0.6.{images_format}"),
                join(alpha_contrast_images_dir_path, f"1.0.{images_format}"),
                join(alpha_contrast_images_dir_path, f"1.4.{images_format}"),
                "Image * alpha=0.6 histogram",
                "No preprocessing image histogram",
                "Image * alpha=1.4 histogram",
                save=join(alpha_contrast_hists_dir_path, f"{dataset_name}.png"),
                set_extreme_values_to_zero=True
            )
        print(f"LOG: Alpha-contrast dataset {alpha_contrast_images_dir_path} were generated!")


    if beta_brightness:
        brightness_images_dir_path = join(gen_images_dir_path, "brightness")
        if not exists(brightness_images_dir_path):
            makedirs(brightness_images_dir_path)

        max_abs_beta, beta_step = 75, 15
        if extended_dataset:
            beta_iterator = np.arange(-max_abs_beta * 7, max_abs_beta * 7 + 1, beta_step)
        else:
            beta_iterator = np.arange(-max_abs_beta, max_abs_beta + 1, beta_step)

        for beta in beta_iterator:
            brightness_shift_image = cv.convertScaleAbs(main_image, alpha=1., beta=beta)
            if any(brightness_shift_image[brightness_shift_image != 255]):
                cv.imwrite(join(brightness_images_dir_path, str(beta) + "." + images_format), brightness_shift_image)

        if plot_hists:
            brightness_hists_dir_path = join(hists_dir_path, "brightness")
            if not exists(brightness_hists_dir_path):
                makedirs(brightness_hists_dir_path)

            plot_hists_for_three_shifts(
                join(brightness_images_dir_path, f"-{max_abs_beta}.{images_format}"),
                join(brightness_images_dir_path, f"0.{images_format}"),
                join(brightness_images_dir_path, f"{max_abs_beta}.{images_format}"),
                f"Image + bias=-{max_abs_beta} histogram",
                "No preprocessing image histogram",
                f"Image + bias={max_abs_beta} histogram",
                save=join(brightness_hists_dir_path, f"{dataset_name}.png"),
                set_extreme_values_to_zero=True
            )
        print(f"LOG: Beta-brightness dataset {brightness_images_dir_path} were generated!")


def shuffle_matrix_np(matrix_np, shuffled_images_dir_path, images_format, shuffled_parts_max_num=10):
    matrix_indexes_list = list()
    shuffled_image_np = matrix_np.copy()
    # Следовало бы это делать на каждой итерации shuffle_num, но тогда структура изображения все равно сохраняется.
    for j in range(matrix_np.shape[0]):
        for i in range(matrix_np.shape[1]):
            matrix_indexes_list.append((j, i))

    for shuffled_parts_num in range(0, shuffled_parts_max_num + 1):
        pixels_to_shuffle_count = matrix_np.shape[0] * matrix_np.shape[1] * shuffled_parts_num // shuffled_parts_max_num
        # list_indexes must be 1-dimensional for np.random.choice
        positions_to_get_in_matrix_indexes_list = np.random.choice(
            len(matrix_indexes_list), size=(pixels_to_shuffle_count // 2, 2), replace=False
        )
        for pos1, pos2 in positions_to_get_in_matrix_indexes_list:
            j1, i1 = matrix_indexes_list[pos1]
            j2, i2 = matrix_indexes_list[pos2]

            (shuffled_image_np[j1][i1], shuffled_image_np[j2][i2]) = (shuffled_image_np[j2][i2], shuffled_image_np[j1][i1])

        counts, bins = np.histogram(shuffled_image_np, bins=range(0, 256, 1))
        cv.imwrite(join(shuffled_images_dir_path, str(shuffled_parts_num) + "." + images_format), shuffled_image_np)


# Source: https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
def generate_datasets_nonlinear(
        images_dir_path,
        dataset_name, images_format,
        extended_dataset=True,
        gamma_contrast=True,
        shuffle=True,
        gaussian_blur=True,
        plot_hists=True
):
    if plot_hists:
        hists_dir_path = join(images_dir_path, "hists")
        if not exists(hists_dir_path):
            makedirs(hists_dir_path)

    if extended_dataset:
        dir_postfix = 2
    else:
        dir_postfix = 1
    pattern = f"{dataset_name}_{dir_postfix}"

    # The image for which the test datasets were generated
    main_image_name = f"{dataset_name}.{images_format}"
    main_image_path = join(images_dir_path, main_image_name)

    main_image = cv.imread(main_image_path)
    if main_image is None:
        print(
            f"Could not open or find the main image {main_image_path} for which the test datasets should be generated!"
        )
        exit(0)

    gen_images_dir_path = join(images_dir_path, f"compare_{pattern}")
    if not exists(gen_images_dir_path):
        makedirs(gen_images_dir_path)

    if gamma_contrast:
        gamma_contrast_images_dir_path = join(gen_images_dir_path, "gamma_contrast")
        if not exists(gamma_contrast_images_dir_path):
            makedirs(gamma_contrast_images_dir_path)

        if extended_dataset:
            gamma_iterator = np.arange(0.1, 2.1, 0.1)
        else:
            gamma_iterator = np.arange(0.5, 1.6, 0.1)

        for gamma in gamma_iterator:
            gamma = round(gamma, 1)
            gamma_shift_image = np.array(255 * (main_image / 255) ** gamma, dtype='uint8')
            cv.imwrite(join(gamma_contrast_images_dir_path, str(gamma) + "." + images_format), gamma_shift_image)

        if plot_hists:
            gamma_contrast_hists_dir_path = join(hists_dir_path, "gamma_contrast")
            if not exists(gamma_contrast_hists_dir_path):
                makedirs(gamma_contrast_hists_dir_path)

            plot_hists_for_three_shifts(
                join(gamma_contrast_images_dir_path, f"0.6.{images_format}"),
                join(gamma_contrast_images_dir_path, f"1.0.{images_format}"),
                join(gamma_contrast_images_dir_path, f"1.4.{images_format}"),
                "Histogram for gamma=0.6",
                "No preprocessing image histogram",
                "Histogram for gamma=1.4",
                save=join(gamma_contrast_hists_dir_path, f"{dataset_name}.png"),
                set_extreme_values_to_zero=False
            )
        print(f"LOG: Gamma-contrast dataset {gamma_contrast_images_dir_path} were generated!")

    if shuffle:
        shuffled_images_dir_path = join(gen_images_dir_path, "shuffle")
        if not exists(shuffled_images_dir_path):
            makedirs(shuffled_images_dir_path)

        shuffle_matrix_np(main_image, shuffled_images_dir_path, images_format, shuffled_parts_max_num=10)

        if plot_hists:
            shuffle_hists_dir_path = join(hists_dir_path, "shuffle")
            if not exists(shuffle_hists_dir_path):
                makedirs(shuffle_hists_dir_path)

            plot_hists_for_three_shifts(
                join(shuffled_images_dir_path, f"0.{images_format}"),
                join(shuffled_images_dir_path, f"5.{images_format}"),
                join(shuffled_images_dir_path, f"10.{images_format}"),
                "Histogram of the original image",
                "Histogram of a half-mixed image",
                "Histogram of a completely mixed image",
                save=join(shuffle_hists_dir_path,  f"{dataset_name}.png"),
                set_extreme_values_to_zero=False
            )
        print(f"LOG: Pixel-shuffled dataset {shuffled_images_dir_path} were generated!")


    if gaussian_blur:
        gaussian_blurry_images_dir_path = join(gen_images_dir_path, "gaussian_blur")
        if not exists(gaussian_blurry_images_dir_path):
            makedirs(gaussian_blurry_images_dir_path)

        if extended_dataset:
            blurry_kernel_size_iterator = np.arange(1, 30, 2)
        else:
            blurry_kernel_size_iterator = np.arange(1, 20, 2)

        cv.imwrite(join(gaussian_blurry_images_dir_path, str(0) + "." + images_format), main_image)

        for blurry_kernel_size in blurry_kernel_size_iterator:
            blurry_image = cv.GaussianBlur(
                main_image,
                (blurry_kernel_size, blurry_kernel_size),
                sigmaX=0,
                borderType=cv.BORDER_REFLECT
            )
            cv.imwrite(join(gaussian_blurry_images_dir_path, str(blurry_kernel_size) + "." + images_format),
                       blurry_image)

        if plot_hists:
            gaussian_blurry_hists_dir_path = join(hists_dir_path, "gaussian_blur")
            if not exists(gaussian_blurry_hists_dir_path):
                makedirs(gaussian_blurry_hists_dir_path)

            plot_hists_for_three_shifts(
                join(gaussian_blurry_images_dir_path, f"0.{images_format}"),
                join(gaussian_blurry_images_dir_path, f"9.{images_format}"),
                join(gaussian_blurry_images_dir_path, f"19.{images_format}"),
                "Histogram of the original image",
                "Histogram of the gaussian blurred image with kernel_size=9",
                "Histogram of the gaussian blurred image with kernel_size=19",
                save=join(gaussian_blurry_hists_dir_path, f"{dataset_name}-.png"),
                set_extreme_values_to_zero=False
            )
        print(f"LOG: Gaussian blur dataset {gaussian_blurry_images_dir_path} were generated!")


def main():
    images_dir_path = join(".", "test_metrics")
    dataset_name = "DR-2D"
    images_format = "png"
    extended_dataset = False

    generate_datasets_linearly(
        images_dir_path,
        dataset_name, images_format,
        extended_dataset=extended_dataset,
        alpha_contrast=False,
        beta_brightness=False,
        plot_hists=True
    )
    generate_datasets_nonlinear(
        images_dir_path,
        dataset_name, images_format,
        extended_dataset=extended_dataset,
        gamma_contrast=False,
        shuffle=True,
        gaussian_blur=False,
        plot_hists=True
    )


if __name__ == '__main__':
    # Code to be run only when run directly
    main()
