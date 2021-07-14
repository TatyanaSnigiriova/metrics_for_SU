import os
import tensorflow as tf
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score, mean_absolute_error


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
    ncc = np.mean(
        np.multiply(
            image1_np - np.mean(image1_np),
            image2_np - np.mean(image2_np)
        )
    ) / (np.std(image1_np) * np.std(image2_np))

    return ncc


def calculate_metrics_for_image(
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
    if calculation_in_uint8:
        metrics_values["MAE"] = mean_absolute_error(HR_file_np_uint8, SR_file_np_uint8)
    else:
        metrics_values["MAE"] = mean_absolute_error(HR_file_np_float32, SR_file_np_float32)
    if log:
        print("\tMAE:", metrics_values["MAE"])

    # MII - Mutual Information Index
    # There are many implementations for calculating the metric of mutual information:
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html
    # mutual_info_classif get only integer values
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html?highlight=mutual_info_score
    # mutual_info_score and mutual_information get both integer and floating-point values
    if calculation_in_uint8:
        metrics_values["MII"] = mutual_info_classif(
            HR_file_np_uint8.reshape(-1, 1),
            SR_file_np_uint8.ravel(),
            discrete_features=True
        )[0]
        # or
        # metrics_value["MII"] = mutual_info_score(HR_file_np_uint8.ravel(), SR_file_np_uint8.ravel())
        # or
        # metrics_value["MII"] = mutual_information_index(HR_file_np_uint8, SR_file_np_uint8, bins=256)
    else:
        metrics_values["MII"] = mutual_info_score(HR_file_np_float32.ravel(), SR_file_np_float32.ravel())
        # or
        # metrics_value["MII"] = mutual_information_index(HR_file_np_float32, SR_file_np_float32, bins=256)
    if log:
        print("\tMutual Info:", metrics_values["MII"])

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

    return metrics_values


def calculate_mean_metrics_for_images(HR_dir_path, SR_dir_path, calculation_in_uint8=False, log=True):
    sess = tf.InteractiveSession()
    list_of_HR_files = sorted(os.listdir(HR_dir_path))
    list_of_SR_files = sorted(os.listdir(SR_dir_path))

    MSE_list = []
    MAE_list = []
    MII_list = []
    PSNR_list = []
    NCC_list = []
    SSIM_list = []
    print()
    for HR_file_name, SR_file_name in zip(list_of_HR_files, list_of_SR_files):
        metrics_values = calculate_metrics_for_image(
            HR_dir_path, HR_file_name,
            SR_dir_path, SR_file_name,
            sess=sess,
            calculation_in_uint8=calculation_in_uint8,
            log=log
        )
        MSE_list.append(metrics_values["MSE"])
        MAE_list.append(metrics_values["MAE"])
        MII_list.append(metrics_values["MII"])
        PSNR_list.append(metrics_values["PSNR"])
        NCC_list.append(metrics_values["NCC"])
        SSIM_list.append(metrics_values["SSIM"])

    mean_metrics_values = dict()
    mean_metrics_values['MSE'] = np.mean(MSE_list)
    mean_metrics_values['MAE'] = np.mean(MAE_list)
    mean_metrics_values['MII'] = np.mean(MII_list)
    mean_metrics_values['PSNR'] = np.mean(PSNR_list)
    mean_metrics_values['NCC'] = np.mean(NCC_list)
    mean_metrics_values['SSIM'] = np.mean(SSIM_list)

    print("\nMean MSE:", round(mean_metrics_values['MSE'], 5))
    print("Mean MAE:", round(mean_metrics_values['MAE'], 5))
    print("Mean MII:", round(mean_metrics_values['MII'], 5))
    print("Mean PSNR: ", round(mean_metrics_values['PSNR'], 5))
    print("Mean NCC:", round(mean_metrics_values['NCC'], 5))
    print("Mean SSIM:", round(mean_metrics_values['SSIM'], 5))

    sess.close()
    return mean_metrics_values
