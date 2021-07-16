from calculate_metrics import *
from os import listdir, makedirs
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
        save_to_dir=None,
        dataset_full_name="",
        show=True,
        separate_plot_for_metric=["PSNR", "MII"]
):
    metrics_shifts_values_ = metrics_shifts_values.copy()
    for metric in separate_plot_for_metric:
        if metric in metrics_shifts_values_.keys():
            fig1, ax1 = plt.subplots()  # Create a figure and an axes.
            fig1.set_size_inches(15, 15)
            ax1.set_xlabel(f'{shift_name.title()} shift', fontsize=20)
            ax1.set_ylabel('Value of the metric', fontsize=20)
            ax1.set_title(f"Changes in the values of metrics by {shift_name}", fontsize=30)
            shifts, values = zip(*sorted(metrics_shifts_values_[metric].items()))
            ax1.plot(shifts, values, 'o-', label=metric)
            max_val = max(*[value for value in values if value != np.inf])
            min_val = min(*values)
            ax1.plot((zero_shift, zero_shift), (min_val, max_val), linestyle="--", label="Max values")
            ax1.legend(fontsize=20)
            plt.grid()
            if show:
                plt.show()
            if save_to_dir:
                plt.savefig(join(save_to_dir, dataset_full_name + f"_{metric}.png"), dpi=100)
            metrics_shifts_values_.pop(metric)

    fig2, ax2 = plt.subplots()  # Create a figure and an axes.
    fig2.set_size_inches(15, 15)
    ax2.set_xlabel(f'{shift_name.title()} shift', fontsize=20)
    ax2.set_ylabel('Value of te metric', fontsize=20)
    ax2.set_title(f"Сhanges in the values of metrics by {shift_name}",fontsize=30)
    max_val = -np.inf
    min_val = np.inf
    for metric in metrics_shifts_values_:
        shifts, values = zip(*sorted(metrics_shifts_values_[metric].items()))
        ax2.plot(shifts, values, 'o-', label=metric)
        max_val = max(*[value for value in values if value != np.inf], max_val)
        min_val = min(*values, min_val)
    ax2.plot((zero_shift, zero_shift), (min_val, max_val), linestyle="--", label="Max values")
    ax2.set_yticks(np.arange(round(min_val, 1), max_val, 0.1))
    ax2.legend(fontsize=20)
    plt.grid()
    if show:
        plt.show()
    if save_to_dir:
        plt.savefig(join(save_to_dir, dataset_full_name + ".png"), dpi=100)


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
        mean_metric_response_left = []
        mean_metric_response_right = []

        for shift in metrics_shifts_values[metric].keys():
            if shift == zero_shift:
                continue
            if shift < zero_shift:
                mean_metric_response_left.append(
                    (metrics_shifts_values[metric][shift] - metric_zero_shift_value) / abs(shift - zero_shift))
            else:
                mean_metric_response_right.append(
                    (metrics_shifts_values[metric][shift] - metric_zero_shift_value) / abs(shift - zero_shift))
        mean_metrics_responses[metric] = dict()
        if round_num == np.inf:
            mean_metrics_responses[metric]["-"] = np.mean(mean_metric_response_left)
            mean_metrics_responses[metric]["+"] = np.mean(mean_metric_response_right)
        else:
            mean_metrics_responses[metric]["-"] = round(np.mean(mean_metric_response_left), round_num)
            mean_metrics_responses[metric]["+"] = round(np.mean(mean_metric_response_right), round_num)
    return mean_metrics_responses


def main():
    images_dir_path = join(".", "test_metrics")
    dataset_name = "M1"
    dir_num = 1
    pattern = f"{dataset_name}_{dir_num}"
    logs_dir_path = join(".", "logs")
    log_name = f"{pattern}.csv"
    plots_dir_path = join(".", "plots")

    # The image for which the test data sets were generated
    main_image_name = f"{dataset_name}.png"
    gen_images_dir_path = join(images_dir_path, f"compare_{pattern}")
    metrics = ["MSE", "MAE", "MII", "PSNR", "NCC", "SSIM"]
    sess = tf.InteractiveSession()

    # brightness
    zero_betta = 0

    brightness_logs_dir_path = join(logs_dir_path, "brightness")
    if not exists(brightness_logs_dir_path):
        makedirs(brightness_logs_dir_path)

    brightness_log_path = join(brightness_logs_dir_path, log_name)
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
        brightness_plots_dir_path = join(plots_dir_path, "brightness")
        if not exists(brightness_plots_dir_path):
            makedirs(brightness_plots_dir_path)
        plot_for_metrics_shifts_values(
            metrics_brightness_values,
            shift_name="brightness",
            zero_shift=zero_betta,
            save_to_dir=brightness_plots_dir_path,
            dataset_full_name=pattern,
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
    contrast_logs_dir_path = join(logs_dir_path, "contrast")
    if not exists(contrast_logs_dir_path):
        makedirs(contrast_logs_dir_path)
    contrast_log_path = join(contrast_logs_dir_path, log_name)
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
        contrast_plots_dir_path = join(plots_dir_path, "contrast")
        if not exists(contrast_plots_dir_path):
            makedirs(contrast_plots_dir_path)
        plot_for_metrics_shifts_values(
            metrics_contrast_values,
            shift_name="contrast",
            zero_shift=zero_alpha,
            save_to_dir=contrast_plots_dir_path,
            dataset_full_name=pattern,
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
    gamma_contrast_logs_dir_path = join(logs_dir_path, "gamma_contrast")
    if not exists(gamma_contrast_logs_dir_path):
        makedirs(gamma_contrast_logs_dir_path)
    gamma_contrast_log_path = join(gamma_contrast_logs_dir_path, pattern)
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
        gamma_contrast_plots_dir_path = join(plots_dir_path, "gamma_contrast")
        if not exists(gamma_contrast_plots_dir_path):
            makedirs(gamma_contrast_plots_dir_path)

        plot_for_metrics_shifts_values(
            metrics_gamma_contrast_values,
            shift_name="gamma contrast",
            zero_shift=zero_gamma,
            save_to_dir=gamma_contrast_plots_dir_path,
            dataset_full_name=pattern,
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
