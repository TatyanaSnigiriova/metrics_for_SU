import os
import tensorflow as tf
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score, mean_absolute_error, normalized_mutual_info_score
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd


def mutual_information(image1_np, image2_np, bins):
    # Source: https://matthew-brett.github.io/teaching/mutual_information.html
    # Mutual information is a metric from the joint (2D) histogram. The metric is
    # high when the signal is highly concentrated in few bins (squares), and low
    # when the signal is spread across many bins (squares).
    #
    # See [http://en.wikipedia.org/wiki/Mutual_information](http://en.wikipedia.org/wiki/Mutual_information)

    hist_2d, x_edges, y_edges = np.histogram2d(
        image1_np.ravel(),
        image2_np.ravel(),
        bins=bins
    )

    # Convert bins counts to probability values
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def NCC(image1_np, image2_np):
    '''
    ncc = np.sum(
              np.multiply(
                  image1_np - np.mean(image1_np),
                  image2_np - np.mean(image2_np)
              ) / np.sqrt(
                  np.multiply(
                      np.sum(np.square(image1_np - np.mean(image1_np))),
                      np.sum(np.square(image2_np - np.mean(image2_np)))
                  )
              )
          )
    '''
    # or
    denominator = (np.std(image1_np) * np.std(image2_np))
    numerator = np.mean(
        np.multiply(
            image1_np - np.mean(image1_np),
            image2_np - np.mean(image2_np)
        )
    )
    if denominator < 1e-5:
        if numerator < 0:
            return - np.inf
        elif numerator < 1e-5:
            return 1
        else:
            return + np.inf
    else:
        return numerator / denominator


# Only for grayscale
def calculate_metrics_for_grayscale_image(
        HR_dir_path, HR_file_name,
        SR_dir_path, SR_file_name,
        sess,
        calculation_in_uint8=False,
        log=True
):
    metrics_values = dict()
    if log:
        print("\n", HR_file_name, SR_file_name)
    HR_file_path = os.path.join(HR_dir_path, HR_file_name)
    SR_file_path = os.path.join(SR_dir_path, SR_file_name)

    HR_file_tf_uint8 = tf.image.decode_jpeg(
        tf.read_file(HR_file_path),
        channels=1
    )
    HR_file_tf_float32 = tf.image.convert_image_dtype(HR_file_tf_uint8, tf.float32)

    SR_file_tf_uint8 = tf.image.decode_jpeg(
        tf.read_file(SR_file_path),
        channels=1
    )
    SR_file_tf_float32 = tf.image.convert_image_dtype(SR_file_tf_uint8, tf.float32)
    HR_file_np_uint8 = sess.run(HR_file_tf_uint8)[:, :, 0]
    SR_file_np_uint8 = sess.run(SR_file_tf_uint8)[:, :, 0]
    HR_file_np_float32 = sess.run(HR_file_tf_float32)[:, :, 0]
    SR_file_np_float32 = sess.run(SR_file_tf_float32)[:, :, 0]

    '''
    METRICS CALCULATIONS
    '''
    # MSE - Mean Squared Error
    # tf.reduce_mean get only float values
    mse_op = tf.reduce_mean(tf.square(HR_file_tf_float32 - SR_file_tf_float32))
    metrics_values["MSE"] = sess.run(mse_op)
    if log:
        print("\tMSE:", metrics_values["MSE"])

    # MAE - Mean Absolute Error
    mae_op = tf.reduce_mean(tf.abs(HR_file_tf_float32 - SR_file_tf_float32))
    metrics_values["MAE"] = sess.run(mae_op)
    if log:
        print("\tMAE:", metrics_values["MAE"])

    # MI - Mutual Information Index
    # There are many implementations for calculating the metric of mutual information:
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html
    # mutual_info_classif get only integer values
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html?highlight=mutual_info_score
    # mutual_info_score and mutual_information get both integer and floating-point values
    # https: // scikit - learn.org / stable / modules / generated / sklearn.metrics.normalized_mutual_info_score.html
    if calculation_in_uint8:
        metrics_values["MI"] = mutual_info_score(HR_file_np_uint8.ravel(), SR_file_np_uint8.ravel())
        metrics_values["NMI"] = normalized_mutual_info_score(HR_file_np_uint8.ravel(), SR_file_np_uint8.ravel())
    else:
        metrics_values["MI"] = mutual_info_score(HR_file_np_float32.ravel(), SR_file_np_float32.ravel())
        metrics_values["NMI"] = normalized_mutual_info_score(HR_file_np_float32.ravel(), SR_file_np_float32.ravel())

    if log:
        print("\tMutual Info:", metrics_values["MI"])
        print("\tNormalized-Mutual Info:", metrics_values["NMI"])

    # PSNR - Peak Signal-To-Noise Ratio
    # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/psnr?hl=ru
    # tf.psnr get both integer and floating-point values
    if calculation_in_uint8:
        psnr_op = tf.image.psnr(HR_file_tf_uint8, SR_file_tf_uint8, max_val=255)
    else:
        psnr_op = tf.image.psnr(HR_file_tf_float32, SR_file_tf_float32, max_val=1.0)
    metrics_values["PSNR"] = sess.run(psnr_op)
    if log:
        print("\tPSNR:", metrics_values["PSNR"])

    # NCC - Normalized cross-correlation
    if calculation_in_uint8:
        metrics_values["NCC"] = NCC(HR_file_np_uint8, SR_file_np_uint8)
    else:
        metrics_values["NCC"] = NCC(HR_file_np_float32, SR_file_np_float32)
    if log:
        print("\tNormalized Cross-Correlation:", metrics_values["NCC"])

    metrics_values["Nx2CC"] = (metrics_values["NCC"] + 1) / 2
    if log:
        print("\tNormalized-Normalized Cross-Correlation:", metrics_values["Nx2CC"])

    # SSIM
    # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/ssim?hl=ru
    # https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/python/ops/image_ops_impl.py
    # Details:
    # - 11x11 Gaussian filter of width 1.5 is used.
    # In this version TF it's only one size of Gaussian filter.
    # - k1 = 0.01, k2 = 0.03 as in the original paper.
    # tf.image.ssim get both integer and floating-point values
    if calculation_in_uint8:
        ssim_op = tf.image.ssim(HR_file_tf_uint8, SR_file_tf_uint8, max_val=255)
    else:
        ssim_op = tf.image.ssim(HR_file_tf_float32, SR_file_tf_float32, max_val=1.0)

    metrics_values["SSIM"] = sess.run(ssim_op)
    if log:
        print("\tSSIM:", metrics_values["SSIM"])

    metrics_values["NSSIM"] = (metrics_values["SSIM"] + 1) / 2
    if log:
        print("\tNSSIM:", metrics_values["NSSIM"])

    return metrics_values


def calculate_mean_metrics_for_grayscale_images(HR_dir_path, SR_dir_path, calculation_in_uint8=False, log=True):
    sess = tf.InteractiveSession()
    list_of_HR_files = sorted(os.listdir(HR_dir_path))
    list_of_SR_files = sorted(os.listdir(SR_dir_path))

    MSE_list = []
    MAE_list = []
    MI_list = []
    NMI_list = []
    PSNR_list = []
    NCC_list = []
    Nx2CC_list = []
    SSIM_list = []
    NSSIM_list = []
    print()
    for HR_file_name, SR_file_name in zip(list_of_HR_files, list_of_SR_files):
        metrics_values = calculate_metrics_for_grayscale_image(
            HR_dir_path, HR_file_name,
            SR_dir_path, SR_file_name,
            sess=sess,
            calculation_in_uint8=calculation_in_uint8,
            log=log
        )

        MSE_list.append(metrics_values["MSE"])
        MAE_list.append(metrics_values["MAE"])
        MI_list.append(metrics_values["MI"])
        NMI_list.append(metrics_values["NMI"])
        PSNR_list.append(metrics_values["PSNR"])
        NCC_list.append(metrics_values["NCC"])
        Nx2CC_list.append(metrics_values["Nx2CC"])
        SSIM_list.append(metrics_values["SSIM"])
        NSSIM_list.append(metrics_values["NSSIM"])

    mean_metrics_values = dict()
    mean_metrics_values['MSE'] = np.mean(MSE_list)
    mean_metrics_values['MAE'] = np.mean(MAE_list)
    mean_metrics_values['MI'] = np.mean(MI_list)
    mean_metrics_values['NMI'] = np.mean(NMI_list)
    mean_metrics_values['PSNR'] = np.mean(PSNR_list)
    mean_metrics_values['NCC'] = np.mean(NCC_list)
    mean_metrics_values['Nx2CC'] = np.mean(Nx2CC_list)
    mean_metrics_values['SSIM'] = np.mean(SSIM_list)
    mean_metrics_values['NSSIM'] = np.mean(NSSIM_list)

    print("\nMean MSE:", round(mean_metrics_values['MSE'], 5))
    print("Mean MAE:", round(mean_metrics_values['MAE'], 5))
    print("Mean MI:", round(mean_metrics_values['MI'], 5))
    print("Mean NMI:", round(mean_metrics_values['NMI'], 5))
    print("Mean PSNR: ", round(mean_metrics_values['PSNR'], 5))
    print("Mean NCC:", round(mean_metrics_values['NCC'], 5))
    print("Mean Nx2CC:", round(mean_metrics_values['Nx2CC'], 5))
    print("Mean SSIM:", round(mean_metrics_values['SSIM'], 5))
    print("Mean NSSIM:", round(mean_metrics_values['NSSIM'], 5))

    sess.close()
    return mean_metrics_values


def calculate_metrics_for_grayscale_images(HR_dir_path, SR_dir_path, csv_path, calculation_in_uint8=False, log=True):
    sess = tf.InteractiveSession()
    list_of_HR_files = sorted(os.listdir(HR_dir_path))
    list_of_SR_files = sorted(os.listdir(SR_dir_path))

    MSE_list = []
    MAE_list = []
    MI_list = []
    NMI_list = []
    PSNR_list = []
    NCC_list = []
    Nx2CC_list = []
    SSIM_list = []
    NSSIM_list = []
    print()
    for HR_file_name, SR_file_name in zip(list_of_HR_files, list_of_SR_files):
        metrics_values = calculate_metrics_for_grayscale_image(
            HR_dir_path, HR_file_name,
            SR_dir_path, SR_file_name,
            sess=sess,
            calculation_in_uint8=calculation_in_uint8,
            log=log
        )
        MSE_list.append(metrics_values["MSE"])
        MAE_list.append(metrics_values["MAE"])
        MI_list.append(metrics_values["MI"])
        NMI_list.append(metrics_values["NMI"])
        PSNR_list.append(metrics_values["PSNR"])
        NCC_list.append(metrics_values["NCC"])
        Nx2CC_list.append(metrics_values["Nx2CC"])
        SSIM_list.append(metrics_values["SSIM"])
        NSSIM_list.append(metrics_values["NSSIM"])

    columns = ["MSE", "MAE", "MI", "NMI", "PSNR", "NCC", "Nx2CC", "SSIM", "NSSIM"]
    data = np.concatenate(
        [[MSE_list], [MAE_list], [MI_list], [NMI_list], [PSNR_list], [NCC_list], [Nx2CC_list], [SSIM_list],
         [NSSIM_list]]).T
    data_pd = pd.DataFrame(data=data, columns=columns)
    data_pd.to_csv(csv_path, index=False)
    sess.close()
