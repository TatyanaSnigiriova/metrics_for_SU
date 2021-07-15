from calculate_metrics import *
from os import listdir
from os.path import join, exists
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
    shifts = list(map(func_to_get_shift, shifted_files_names))
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


def plot_for_metrics_shifts_values(
        metrics_shifts_values,
        shift_name,
        zero_shift,
        save_to_dir=None, dir_num="",
        show=True
):
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
            plt.savefig(join(save_to_dir, shift_name + f"_{dir_num}" + "PSNR" + ".png"), dpi=100)
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
        plt.savefig(join(save_to_dir, shift_name + f"{dir_num}.png"), dpi=100)


def csv_writer_for_metrics_shifts_values(metrics_shifts_values, output_csv_path):
    w = csv.writer(open(output_csv_path, "w"))
    shifts = sorted(metrics_shifts_values[list(metrics_shifts_values.keys())[0]].keys())
    w.writerow(["Metrics\Shifts", *shifts])
    for metric in metrics_shifts_values.keys():
        shifts, values = zip(*sorted(metrics_shifts_values[metric].items()))
        w.writerow([metric, *values])


def csv_reader_for_metrics_shifts_values(input_csv_name, func_to_get_shift):
    reader = csv.reader(open(input_csv_name, "r", newline='\n'))
    metrics_shifts_values = dict()
    line_num = 0
    for line in reader:
        if line_num == 0:
            shifts = line.copy()
            shifts.pop(0)
            shifts = list(map(func_to_get_shift, shifts))
        else:
            values = line.copy()
            metric = values[0]
            values.pop(0)
            values = list(map(float, values))
            metrics_shifts_values[metric] = dict(zip(shifts, values))
        line_num += 1

    return metrics_shifts_values


def calculate_mean_shift_response_for_metrics(metrics_shifts_values, zero_shift, round_num=np.inf):
    mean_metrics_responses = dict()
    for metric in metrics_shifts_values.keys():
        metric_zero_shift_value = metrics_shifts_values[metric][zero_shift]
        mean_metric_response = []
        for shift in metrics_shifts_values[metric].keys():
            if shift == zero_shift:
                continue
            mean_metric_response.append(abs(metric_zero_shift_value - metrics_shifts_values[metric][shift]))
        if round_num == np.inf:
            mean_metrics_responses[metric] = np.mean(mean_metric_response)
        else:
            mean_metrics_responses[metric] = round(np.mean(mean_metric_response), round_num)
    return mean_metrics_responses


def main():
    images_dir_path = join(".", "test_metrics")
    plots_dir_path = join(".", "plots")
    logs_dir_path = join(".", "logs")
    dir_num = 1

    # The image for which the test data sets were generated
    main_image_name = "HR.png"

    gen_images_dir_path = join(images_dir_path, f"compare_{dir_num}")

    metrics = ["MSE", "MAE", "MII", "PSNR", "NCC", "SSIM"]
    sess = tf.InteractiveSession()

    # brightness
    zero_betta = 0
    brightness_log_name = f"brightness_{dir_num}.csv"
    brightness_log_path = join(logs_dir_path, brightness_log_name)
    if not exists(brightness_log_path):
        print("Calculate, plot and write metrics values for brightness shift")
        brightness_images_dir_path = join(gen_images_dir_path, "brightness")
        metrics_brightness_values = get_metrics_values_for_image_shifts(
            images_dir_path, main_image_name,
            brightness_images_dir_path,
            get_brightness,
            zero_betta,  # betta = 0
            metrics,
            sess,
        )
        plot_for_metrics_shifts_values(
            metrics_brightness_values,
            shift_name="brightness",
            zero_shift=zero_betta,
            save_to_dir=plots_dir_path,
            dir_num=dir_num,
            show=False
        )
        csv_writer_for_metrics_shifts_values(metrics_brightness_values, brightness_log_path)
    else:
        metrics_brightness_values = csv_reader_for_metrics_shifts_values(brightness_log_path, int)
    mean_metrics_responses_to_brightness = calculate_mean_shift_response_for_metrics(
        metrics_brightness_values,
        zero_betta,
        round_num=4
    )
    print("Mean metrics responses to brightness:")
    print(mean_metrics_responses_to_brightness)

    # contrast
    zero_alpha = 1.
    contrast_log_name = f"contrast_{dir_num}.csv"
    contrast_log_path = join(logs_dir_path, contrast_log_name)
    if not exists(contrast_log_path):
        print("Calculate, plot and write metrics values for contrast shift")
        contrast_images_dir_path = join(gen_images_dir_path, "contrast")
        metrics_contrast_values = get_metrics_values_for_image_shifts(
            images_dir_path, main_image_name,
            contrast_images_dir_path,
            get_contrast,
            zero_alpha,  # alpha = 1.
            metrics,
            sess,
        )
        plot_for_metrics_shifts_values(
            metrics_contrast_values,
            shift_name="contrast",
            zero_shift=zero_alpha,
            save_to_dir=plots_dir_path,
            dir_num=dir_num,
            show=False
        )
        csv_writer_for_metrics_shifts_values(metrics_contrast_values, contrast_log_path)
    else:
        metrics_contrast_values = csv_reader_for_metrics_shifts_values(contrast_log_path, float)
    mean_metrics_responses_to_contrast = calculate_mean_shift_response_for_metrics(
        metrics_contrast_values,
        zero_alpha,
        round_num=4
    )
    print("Mean metrics responses to contrast:")
    print(mean_metrics_responses_to_contrast)

    # gamma contrast
    zero_gamma = 1.
    gamma_contrast_log_name = f"gamma_contrast_{dir_num}.csv"
    gamma_contrast_log_path = join(logs_dir_path, gamma_contrast_log_name)
    if not exists(gamma_contrast_log_path):
        print("Calculate, plot and write metrics values for gamma-contrast shift")
        gamma_contrast_images_dir_path = join(gen_images_dir_path, "gamma_contrast")
        metrics_gamma_contrast_values = get_metrics_values_for_image_shifts(
            images_dir_path, main_image_name,
            gamma_contrast_images_dir_path,
            get_contrast,
            zero_gamma,  # gamma = 1.
            metrics,
            sess,
        )
        plot_for_metrics_shifts_values(
            metrics_gamma_contrast_values,
            shift_name="gamma contrast",
            zero_shift=zero_gamma,
            save_to_dir=plots_dir_path,
            dir_num=dir_num,
            show=False
        )
        csv_writer_for_metrics_shifts_values(metrics_gamma_contrast_values, gamma_contrast_log_path)
    else:
        metrics_gamma_contrast_values = csv_reader_for_metrics_shifts_values(gamma_contrast_log_path, float)
    mean_metrics_responses_to_gamma_contrast = calculate_mean_shift_response_for_metrics(
        metrics_gamma_contrast_values,
        zero_gamma,
        round_num=4
    )
    print("Mean metrics responses to gamma contrast:")
    print(mean_metrics_responses_to_gamma_contrast)


if __name__ == '__main__':
    # Code to be run only when run directly
    main()
