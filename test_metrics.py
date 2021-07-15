from calculate_metrics import *
from os import listdir, makedirs
from os.path import isfile, join, exists
from matplotlib import pyplot as plt
import numpy as np
import csv


def get_brightness(file_name):
    return int(file_name[:file_name.rfind(".")])


def get_contrast(file_name):
    return float(file_name[:file_name.rfind(".")])


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

    if zero_shift not in shifts:
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


def plot_for_metrics_shifts_values(metrics_shifts_values, shift_name, zero_shift, save_to_dir=None, show=True):
    if "PSNR" in metrics_shifts_values.keys():
        fig1, ax1 = plt.subplots()  # Create a figure and an axes.
        fig1.set_size_inches(15, 15)
        ax1.set_xlabel(f'{shift_name.title()} shift')
        ax1.set_ylabel('Value of the metric')
        ax1.set_title(f"Changes in the values of metrics by {shift_name}")
        shifts, values = zip(*sorted(metrics_shifts_values["PSNR"].items()))
        ax1.plot(shifts, values, 'o-', label="PSNR")
        max_val = max(*[value for value in values if value != np.inf])
        min_val = min(*values)
        ax1.plot((zero_shift, zero_shift), (min_val, max_val), linestyle="--", label="Max values")
        ax1.legend()
        plt.grid()
        if show:
            plt.show()
        if save_to_dir:
            plt.savefig(join(save_to_dir, shift_name + "_" + "PSNR" + ".png"), dpi=100)
        metrics_shifts_values.pop("PSNR")

    fig2, ax2 = plt.subplots()  # Create a figure and an axes.
    fig2.set_size_inches(15, 15)
    ax2.set_xlabel(f'{shift_name.title()} shift')
    ax2.set_ylabel('Value of te metric')
    ax2.set_title(f"Сhanges in the values of metrics by {shift_name}")
    max_val = -np.inf
    min_val = np.inf
    for metric in metrics_shifts_values:
        shifts, values = zip(*sorted(metrics_shifts_values[metric].items()))
        ax2.plot(shifts, values, 'o-', label=metric)
        max_val = max(*[value for value in values if value != np.inf], max_val)
        min_val = min(*values, min_val)
    ax2.plot((zero_shift, zero_shift), (min_val, max_val), linestyle="--", label="Max values")
    ax2.set_yticks(np.arange(round(min_val, 1), max_val, 0.1))
    ax2.legend()
    plt.grid()
    if show:
        plt.show()
    if save_to_dir:
        plt.savefig(join(save_to_dir, shift_name + ".png"), dpi=100)


def csv_writer_for_metrics_shifts_values(metrics_shifts_values, output_csv_name):
    w = csv.writer(open(output_csv_name, "w"))
    shifts = sorted(metrics_shifts_values[list(metrics_shifts_values.keys())[0]].keys())
    w.writerow(["Metrics\Shifts", *shifts])
    for metric in metrics_shifts_values.keys():
        shifts, values = zip(*sorted(metrics_shifts_values[metric].items()))
        w.writerow([metric, *values])


def main():
    images_dir_path = join(".", "test_metrics")
    plots_dir_path = join(".", "plots")
    logs_dir_path = join(".", "logs")

    # The image for which the test data sets were generated
    main_image_name = "HR.png"

    gen_images_dir_path = join(images_dir_path, "compare")

    metrics = ["MSE", "MAE", "MII", "PSNR", "NCC", "SSIM"]
    sess = tf.InteractiveSession()

    # brightness
    brightness_images_dir_path = join(gen_images_dir_path, "brightness")
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
        zero_shift=0,
        save_to_dir=plots_dir_path,
        show=False
    )
    csv_writer_for_metrics_shifts_values(metrics_brightness_values, join(logs_dir_path, "brightness.csv"))

    # contrast
    contrast_images_dir_path = join(gen_images_dir_path, "contrast")
    metrics_contrast_values = get_metrics_values_for_image_shifts(
        images_dir_path, main_image_name,
        contrast_images_dir_path,
        get_contrast,
        1.,  # alpha = 1.
        metrics,
        sess,
    )
    plot_for_metrics_shifts_values(
        metrics_contrast_values,
        shift_name="contrast",
        zero_shift=1.,
        save_to_dir=plots_dir_path,
        show=False
    )
    csv_writer_for_metrics_shifts_values(metrics_contrast_values, join(logs_dir_path, "contrast.csv"))


if __name__ == '__main__':
    # Code to be run only when run directly
    main()
