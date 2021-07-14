from calculate_metrics import *
from os import listdir, makedirs
from os.path import isfile, join, exists
from matplotlib import pyplot as plt
import numpy as np


def get_brightness(file_name, beta=0):
    return int(file_name[:file_name.rfind(".")]) + beta


def get_metrics_values_for_image_shifts(
        main_image_dir_path, main_image_name,
        shifted_images_dir_path,
        func_to_get_shift,
        zero_shift,
        metrics,
        sess,
        calculation_in_uint8=False,
        log=False
):
    shifted_files_names = listdir(shifted_images_dir_path)
    shifts = [func_to_get_shift(file_name) for file_name in shifted_files_names]
    shifts.append(zero_shift)  # For the main image
    shifts.sort()

    metrics_shifts_values = dict()
    for metric in metrics:
        metrics_shifts_values[metric] = dict.fromkeys(shifts)

    for shifted_file_name in shifted_files_names:
        metrics_values = calculate_metrics_for_image(
            main_image_dir_path, main_image_name,
            shifted_images_dir_path, shifted_file_name,
            sess=sess,
            calculation_in_uint8=calculation_in_uint8,
            log=log
        )
        shift = func_to_get_shift(shifted_file_name)
        for metric in metrics_values.keys():
            metrics_shifts_values[metric][shift] = metrics_values[metric]

    # Compare main image with the same image
    metrics_values = calculate_metrics_for_image(
        main_image_dir_path, main_image_name,
        main_image_dir_path, main_image_name,
        sess=sess,
        calculation_in_uint8=False,
        log=False
    )
    for metric in metrics_values.keys():
        metrics_shifts_values[metric][zero_shift] = metrics_values[metric]
    return metrics_shifts_values


def plot_for_metrics_shifts_values(metrics_shifts_values, shift_name, save_to_dir=None, show=True):
    fig, ax = plt.subplots()  # Create a figure and an axes.
    fig.set_size_inches(15, 15)
    ax.set_xlabel(f'{shift_name.title()} shift')
    ax.set_ylabel('Value of te metric')
    ax.set_title(f"Сhanges in the values of metrics by {shift_name}")
    ax.legend()  # Add a legend.
    max_val = 0
    for metric in metrics_shifts_values:
        shifts, values = zip(*sorted(metrics_shifts_values[metric].items()))
        ax.plot(shifts, values, 'o-', label=metric)
        max_val = max(max([value for value in values if value != np.inf]), max_val)
    ax.plot((0, 0), (0, max_val), linestyle="--", label="Max values")
    ax.legend()
    if show:
        plt.show()
    if save_to_dir:
        plt.savefig(join(save_to_dir, shift_name + ".png"), dpi=100)


def main():
    images_dir_path = join(".", "test_metrics")
    plots_dir_path = join(".", "plots")
    # The image for which the test data sets were generated
    main_image_name = "HR.png"

    gen_images_dir_path = join(images_dir_path, "compare")
    brightness_images_dir_path = join(gen_images_dir_path, "brightness")

    metrics = ["MSE", "MAE", "MII", "PSNR", "NCC", "SSIM"]
    sess = tf.InteractiveSession()

    # brightness
    metrics_brightness_values = get_metrics_values_for_image_shifts(
        images_dir_path, main_image_name,
        brightness_images_dir_path,
        get_brightness,
        0,  # betta = 0
        metrics,
        sess,
    )
    plot_for_metrics_shifts_values(
        metrics_brightness_values,
        shift_name="brightness",
        save_to_dir=plots_dir_path,
        show=False
    )

    # contrast
    contrast_images_dir_path = join(gen_images_dir_path, "contrast")
    # ...


if __name__ == '__main__':
    # Code to be run only when run directly
    main()
