from calculate_metrics import *
from os import listdir, makedirs
from os.path import join, exists
from matplotlib import pyplot as plt
import numpy as np
import csv

diap_metrics = {
    "MSE": {"max": 0, "min": 1},
    "MAE": {"max": 0, "min": 1},
    "MI": {"max": np.inf, "min": 0},
    "NMI": {"max": 1, "min": 0},
    "NCC": {"max": 1, "min": -1},
    "Nx2CC": {"max": 1, "min": 0},
    "SSIM": {"max": 1, "min": -1},
    "NSSIM": {"max": 1, "min": 0},
}

not_normalized_metrics = ["PSNR", "MI", "NCC", "SSIM"]
minimized_metrics = ["MAE", "MSE"]

def get_int_value_from_file_name(file_name):
    return int(file_name[:file_name.rfind(".")])


def get_float_from_file_name(file_name):
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
        metrics_values = calculate_metrics_for_grayscale_image(
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
        metrics_values = calculate_metrics_for_grayscale_image(
            main_image_dir_path, main_image_name,
            main_image_dir_path, main_image_name,
            sess=sess,
            calculation_in_uint8=calculation_in_uint8,
            log=log
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
):
    global not_normalized_metrics, minimized_metrics

    metrics_shifts_values_ = metrics_shifts_values.copy()
    for metric in not_normalized_metrics:
        if metric in metrics_shifts_values_.keys():
            fig1, ax1 = plt.subplots()  # Create a figure and an axes.
            fig1.set_size_inches(15, 15)
            ax1.set_xlabel(f'{shift_name.title()} shift', fontsize=20)
            ax1.set_ylabel('The metric values', fontsize=20)
            ax1.set_title(f"The response of metrics to the {shift_name} shift", fontsize=30)
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
    ax2.set_ylabel('The metric values', fontsize=20)
    ax2.set_title(f"The response of metrics to the {shift_name} shift", fontsize=30)
    for metric in metrics_shifts_values_.keys():
        shifts, values = zip(*sorted(metrics_shifts_values_[metric].items()))
        if metric not in minimized_metrics:
            values = 1 - np.array(values)
            metric = "1 - " + metric
        ax2.plot(shifts, values, 'o-', label=metric)
        max_val = max(*values, max_val)
    ax2.plot((zero_shift, zero_shift), (0, 1), linestyle="--", label="Max values")
    ax2.set_ylim((-0.1, 1.1))
    ax2.set_yticks(np.arange(0, 1.01, 0.05))
    ax2.legend(fontsize=20)
    plt.grid()
    if show:
        plt.show()
    if save_to_dir:
        plt.savefig(join(save_to_dir, f"{dataset_full_name}.png"), dpi=100)


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
    metrics_responses = dict()
    global not_normalized_metrics, minimized_metrics

    for metric in metrics_shifts_values.keys():
        if metric == "PSNR":
            continue
        if metric == "MI":
            diap_metrics[metric]["max"] = max(metrics_shifts_values[metric].values())
        metric_shifts_values = metrics_shifts_values[metric]
        metric_response_left = []
        metric_response_right = []
        shifts_list = sorted(metric_shifts_values.keys())
        zero_shift_i = shifts_list.index(zero_shift)

        if zero_shift_i != 0:
            if metric in minimized_metrics:
                left_max_S = (diap_metrics[metric]["min"] - diap_metrics[metric]["max"])/2
                left_max_S += (diap_metrics[metric]["min"] - diap_metrics[metric]["max"]) * (zero_shift_i - 1)
            else:
                left_max_S = (diap_metrics[metric]["max"] - diap_metrics[metric]["min"]) * zero_shift_i\
                             - (diap_metrics[metric]["max"] - diap_metrics[metric]["min"])/2
        else:
            left_max_S = 0

        if len(shifts_list) - zero_shift_i != 1:
            if metric in minimized_metrics:
                right_max_S = (diap_metrics[metric]["min"] - diap_metrics[metric]["max"]) / 2
                right_max_S += (diap_metrics[metric]["min"] - diap_metrics[metric]["max"]) * (len(shifts_list) - zero_shift_i - 2)
            else:
                right_max_S = (diap_metrics[metric]["max"] - diap_metrics[metric]["min"]) * (len(shifts_list) - zero_shift_i - 1) \
                             - (diap_metrics[metric]["max"] - diap_metrics[metric]["min"]) / 2
        else:
            right_max_S = 0

        for i in range(len(shifts_list) - 1):
            if i < zero_shift_i:
                if minimized_metrics:
                    metric_response_left.append(
                        (
                            metrics_shifts_values[metric][shifts_list[i]] +
                            metrics_shifts_values[metric][shifts_list[i + 1]] -
                            2 * diap_metrics[metric]["max"]
                        ) / 2
                    )
                else:
                    metric_response_left.append(
                        (
                                2 * diap_metrics[metric]["max"] -
                                metrics_shifts_values[metric][shifts_list[i]] -
                                metrics_shifts_values[metric][shifts_list[i + 1]]

                        ) / 2
                    )
            else:
                if minimized_metrics:
                    metric_response_right.append(
                        (
                                metrics_shifts_values[metric][shifts_list[i]] +
                                metrics_shifts_values[metric][shifts_list[i + 1]] -
                                2 * diap_metrics[metric]["max"]
                        ) / 2
                    )
                else:
                    metric_response_right.append(
                        (
                                2 * diap_metrics[metric]["max"] -
                                metrics_shifts_values[metric][shifts_list[i]] -
                                metrics_shifts_values[metric][shifts_list[i + 1]]

                        ) / 2
                    )
        metrics_responses[metric] = dict()
        if round_num == np.inf:
            metrics_responses[metric]["-"] = np.sum(metric_response_left) / left_max_S
            metrics_responses[metric]["+"] = np.sum(metric_response_right) / right_max_S
        else:
            metrics_responses[metric]["-"] = round(np.sum(metric_response_left) / left_max_S, round_num)
            metrics_responses[metric]["+"] = round(np.sum(metric_response_right) / right_max_S, round_num)
    return metrics_responses


def main():
    images_dir_path = join(".", "test_metrics")
    dataset_name = "DR-2D"
    dir_num = 1
    pattern = f"{dataset_name}_{dir_num}"
    logs_dir_path = join(".", "logs")
    log_name = f"{pattern}.csv"
    plots_dir_path = join(".", "plots")
    replot = False
    calculation_in_uint8 = True
    log = False

    # The image for which the test data sets were generated
    main_image_name = f"{dataset_name}.png"
    gen_images_dir_path = join(images_dir_path, f"compare_{pattern}")
    metrics = ["MSE", "MAE", "MI", "NMI", "PSNR", "NCC", "Nx2CC", "SSIM", "NSSIM"]
    sess = tf.InteractiveSession()

    # brightness
    zero_betta = 0
    brightness_dirs_name = "brightness"
    brightness_logs_dir_path = join(logs_dir_path, brightness_dirs_name)
    if not exists(brightness_logs_dir_path):
        makedirs(brightness_logs_dir_path)
    brightness_log_path = join(brightness_logs_dir_path, log_name)
    if not exists(brightness_log_path):
        print("Calculate, plot and write metrics values for brightness shift")
        brightness_images_dir_path = join(gen_images_dir_path, brightness_dirs_name)
        brightness_metrics_values = get_metrics_values_for_image_shifts(
            images_dir_path, main_image_name,
            brightness_images_dir_path,
            get_int_value_from_file_name,
            zero_betta,  # betta = 0
            metrics,
            sess,
            calculation_in_uint8,
            log
        )
        brightness_plots_dir_path = join(plots_dir_path, brightness_dirs_name)
        if not exists(brightness_plots_dir_path):
            makedirs(brightness_plots_dir_path)
        plot_for_metrics_shifts_values(
            brightness_metrics_values,
            shift_name="brightness",
            zero_shift=zero_betta,
            save_to_dir=brightness_plots_dir_path,
            dataset_full_name=pattern,
            show=False
        )
        csv_writer_for_metrics_shifts_values(brightness_metrics_values, brightness_log_path)
    else:
        brightness_metrics_values = csv_reader_for_metrics_shifts_values(brightness_log_path, int)

    if replot:
        brightness_plots_dir_path = join(plots_dir_path, brightness_dirs_name)
        plot_for_metrics_shifts_values(
            brightness_metrics_values,
            shift_name="brightness",
            zero_shift=zero_betta,
            save_to_dir=brightness_plots_dir_path,
            dataset_full_name=pattern,
            show=False
        )
    mean_metrics_responses_to_brightness = calculate_mean_shift_response_for_metrics(
        brightness_metrics_values,
        zero_betta,
        round_num=4,
    )
    print("Mean metrics responses to brightness:")
    print(mean_metrics_responses_to_brightness)

    # alpha contrast
    zero_alpha = 1.
    alpha_contrast_dirs_name = "alpha_contrast"
    alpha_contrast_logs_dir_path = join(logs_dir_path, alpha_contrast_dirs_name)
    if not exists(alpha_contrast_logs_dir_path):
        makedirs(alpha_contrast_logs_dir_path)
    alpha_contrast_log_path = join(alpha_contrast_logs_dir_path, log_name)
    if not exists(alpha_contrast_log_path):
        print("Calculate, plot and write metrics values for contrast shift")
        alpha_contrast_images_dir_path = join(gen_images_dir_path, alpha_contrast_dirs_name)
        alpha_contrast_metrics_values = get_metrics_values_for_image_shifts(
            images_dir_path, main_image_name,
            alpha_contrast_images_dir_path,
            get_float_from_file_name,
            zero_alpha,  # alpha = 1.
            metrics,
            sess,
            calculation_in_uint8,
            log
        )
        alpha_contrast_plots_dir_path = join(plots_dir_path, alpha_contrast_dirs_name)
        if not exists(alpha_contrast_plots_dir_path):
            makedirs(alpha_contrast_plots_dir_path)
        plot_for_metrics_shifts_values(
            alpha_contrast_metrics_values,
            shift_name="contrast",
            zero_shift=zero_alpha,
            save_to_dir=alpha_contrast_plots_dir_path,
            dataset_full_name=pattern,
            show=False
        )
        csv_writer_for_metrics_shifts_values(alpha_contrast_metrics_values, alpha_contrast_log_path)
    else:
        alpha_contrast_metrics_values = csv_reader_for_metrics_shifts_values(alpha_contrast_log_path, float)

    if replot:
        alpha_contrast_plots_dir_path = join(plots_dir_path, alpha_contrast_dirs_name)
        plot_for_metrics_shifts_values(
            alpha_contrast_metrics_values,
            shift_name="contrast",
            zero_shift=zero_alpha,
            save_to_dir=alpha_contrast_plots_dir_path,
            dataset_full_name=pattern,
            show=False
        )
    mean_metrics_responses_to_alpha_contrast = calculate_mean_shift_response_for_metrics(
        alpha_contrast_metrics_values,
        zero_alpha,
        round_num=4,
    )
    print("Mean metrics responses to alpha-contrast:")
    print(mean_metrics_responses_to_alpha_contrast)

    # gamma contrast
    zero_gamma = 1.
    gamma_contrast_dirs_name = "gamma_contrast"
    gamma_contrast_logs_dir_path = join(logs_dir_path, gamma_contrast_dirs_name)
    if not exists(gamma_contrast_logs_dir_path):
        makedirs(gamma_contrast_logs_dir_path)
    gamma_contrast_log_path = join(gamma_contrast_logs_dir_path, pattern)
    if not exists(gamma_contrast_log_path):
        print("Calculate, plot and write metrics values for gamma-contrast shift")
        gamma_contrast_images_dir_path = join(gen_images_dir_path, gamma_contrast_dirs_name)
        gamma_contrast_metrics_values = get_metrics_values_for_image_shifts(
            images_dir_path, main_image_name,
            gamma_contrast_images_dir_path,
            get_float_from_file_name,
            zero_gamma,  # gamma = 1.
            metrics,
            sess,
            calculation_in_uint8,
            log
        )
        gamma_contrast_plots_dir_path = join(plots_dir_path, gamma_contrast_dirs_name)
        if not exists(gamma_contrast_plots_dir_path):
            makedirs(gamma_contrast_plots_dir_path)

        plot_for_metrics_shifts_values(
            gamma_contrast_metrics_values,
            shift_name="gamma contrast",
            zero_shift=zero_gamma,
            save_to_dir=gamma_contrast_plots_dir_path,
            dataset_full_name=pattern,
            show=False
        )
        csv_writer_for_metrics_shifts_values(gamma_contrast_metrics_values, gamma_contrast_log_path)
    else:
        gamma_contrast_metrics_values = csv_reader_for_metrics_shifts_values(gamma_contrast_log_path, float)

    if replot:
        gamma_contrast_plots_dir_path = join(plots_dir_path, gamma_contrast_dirs_name)
        plot_for_metrics_shifts_values(
            gamma_contrast_metrics_values,
            shift_name="gamma contrast",
            zero_shift=zero_gamma,
            save_to_dir=gamma_contrast_plots_dir_path,
            dataset_full_name=pattern,
            show=False
        )
    mean_metrics_responses_to_gamma_contrast = calculate_mean_shift_response_for_metrics(
        gamma_contrast_metrics_values,
        zero_gamma,
        round_num=4,
    )
    print("Mean metrics responses to gamma contrast:")
    print(mean_metrics_responses_to_gamma_contrast)

    # shuffle
    zero_shuffle = 0
    shuffle_dirs_name = "shuffle"
    shuffle_logs_dir_path = join(logs_dir_path, shuffle_dirs_name)
    if not exists(shuffle_logs_dir_path):
        makedirs(shuffle_logs_dir_path)
    shuffle_log_path = join(shuffle_logs_dir_path, pattern)
    if not exists(shuffle_log_path):
        print("Calculate, plot and write metrics values for pixel-shuffled images")
        shuffle_images_dir_path = join(gen_images_dir_path, shuffle_dirs_name)
        shuffle_metrics_values = get_metrics_values_for_image_shifts(
            images_dir_path, main_image_name,
            shuffle_images_dir_path,
            get_int_value_from_file_name,
            zero_shuffle,  # gamma = 1.
            metrics,
            sess,
            calculation_in_uint8,
            log
        )
        shuffle_plots_dir_path = join(plots_dir_path, shuffle_dirs_name)
        if not exists(shuffle_plots_dir_path):
            makedirs(shuffle_plots_dir_path)

        plot_for_metrics_shifts_values(
            shuffle_metrics_values,
            shift_name="shuffle",
            zero_shift=zero_shuffle,
            save_to_dir=shuffle_plots_dir_path,
            dataset_full_name=pattern,
            show=False
        )
        csv_writer_for_metrics_shifts_values(shuffle_metrics_values, shuffle_log_path)
    else:
        shuffle_metrics_values = csv_reader_for_metrics_shifts_values(shuffle_log_path, float)

    if replot:
        shuffle_plots_dir_path = join(plots_dir_path, shuffle_dirs_name)
        plot_for_metrics_shifts_values(
            shuffle_metrics_values,
            shift_name="shuffle",
            zero_shift=zero_shuffle,
            save_to_dir=shuffle_plots_dir_path,
            dataset_full_name=pattern,
            show=False
        )
    mean_metrics_responses_to_shuffle = calculate_mean_shift_response_for_metrics(
        shuffle_metrics_values,
        zero_shuffle,
        round_num=4,
    )
    print("Mean metrics responses to pixel-shuffle:")
    print(mean_metrics_responses_to_shuffle)

    # gaussian blur
    zero_blur = 0
    gaussian_blur_dirs_name = "gaussian_blur"
    gaussian_blur_logs_dir_path = join(logs_dir_path, gaussian_blur_dirs_name)
    if not exists(gaussian_blur_logs_dir_path):
        makedirs(gaussian_blur_logs_dir_path)
    gaussian_blur_log_path = join(gaussian_blur_logs_dir_path, pattern)
    if not exists(gaussian_blur_log_path):
        print("Calculate, plot and write metrics values for gaussian blurry images")
        gaussian_blurry_images_dir_path = join(gen_images_dir_path, gaussian_blur_dirs_name)
        gaussian_blur_metrics_values = get_metrics_values_for_image_shifts(
            images_dir_path, main_image_name,
            gaussian_blurry_images_dir_path,
            get_int_value_from_file_name,
            zero_blur,  # gamma = 1.
            metrics,
            sess,
            calculation_in_uint8,
            log
        )
        gaussian_blur_plots_dir_path = join(plots_dir_path, gaussian_blur_dirs_name)
        if not exists(gaussian_blur_plots_dir_path):
            makedirs(gaussian_blur_plots_dir_path)

        plot_for_metrics_shifts_values(
            gaussian_blur_metrics_values,
            shift_name="gaussian blur",
            zero_shift=zero_blur,
            save_to_dir=gaussian_blur_plots_dir_path,
            dataset_full_name=pattern,
            show=False
        )
        csv_writer_for_metrics_shifts_values(gaussian_blur_metrics_values, gaussian_blur_log_path)
    else:
        gaussian_blur_metrics_values = csv_reader_for_metrics_shifts_values(gaussian_blur_log_path, float)

    if replot:
        gaussian_blur_plots_dir_path = join(plots_dir_path, gaussian_blur_dirs_name)
        plot_for_metrics_shifts_values(
            gaussian_blur_metrics_values,
            shift_name="gaussian blur",
            zero_shift=zero_blur,
            save_to_dir=gaussian_blur_plots_dir_path,
            dataset_full_name=pattern,
            show=False
        )
    mean_metrics_responses_to_gaussian_blur = calculate_mean_shift_response_for_metrics(
        gaussian_blur_metrics_values,
        zero_blur,
        round_num=4,
    )
    print("Mean metrics responses to gaussian_blur:")
    print(mean_metrics_responses_to_gaussian_blur)


if __name__ == '__main__':
    # Code to be run only when run directly
    main()
