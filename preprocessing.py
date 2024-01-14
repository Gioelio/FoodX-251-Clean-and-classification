import cv2 as cv
import numpy as np


def noise_removal(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gradient = cv.Laplacian(gray, ddepth=3)
    mean, std = cv.meanStdDev(gradient)

    if std[0, 0] > 100:
        image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
        ksize = 15

        image[:, :, 1] = cv.GaussianBlur(image[:, :, 1], (ksize, ksize), 9, None, 9)
        image[:, :, 2] = cv.GaussianBlur(image[:, :, 2], (ksize, ksize), 9, None, 9)

        image = cv.cvtColor(image, cv.COLOR_YCrCb2BGR)
        image = cv.GaussianBlur(image, (3, 3), 5, None, 5)
        image = cv.bilateralFilter(image, 7, 140, 5)
    return image


def white_balance(image, threshold1=0.5, threshold2=0.5):
    sat = (np.max(image, axis=2) - np.min(image, axis=2)) / np.max(image, axis=2)

    sat_threshold = np.quantile(sat[~np.isnan(sat)].flatten(), threshold2)

    wb = cv.xphoto.createGrayworldWB()
    wb.setSaturationThreshold(sat_threshold)
    image = wb.balanceWhite(image)

    return image


def local_gamma_correction(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    mask = image[:, :, 0]
    mask = 255 - cv.bilateralFilter(mask, 11, 15, 60)
    y_channel = image[:, :, 0]
    y_channel = 255. * (y_channel / 255.) ** (2. ** ((128. - mask) / 128.))
    image[:, :, 0] = y_channel

    image = cv.cvtColor(image, cv.COLOR_YCrCb2BGR)
    return image


def contrast_stretch(image, ignore_p=0, min_val=0, max_val=255):
    image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    channel = 0
    maximum = np.percentile(image[:, :, channel], 100 - ignore_p)
    minimum = np.percentile(image[:, :, channel], ignore_p)

    if maximum <= minimum:
        maximum = image[:, :, channel].max()
        minimum = image[:, :, channel].min()

    image[:, :, 0][image[:, :, 0] < minimum] = minimum
    image[:, :, 0][image[:, :, 0] > maximum] = maximum

    image[:, :, channel] = (image[:, :, channel] - minimum) * float((max_val - min_val) / float(maximum - minimum))

    image = cv.cvtColor(image, cv.COLOR_YCrCb2BGR)
    return image


def gamma_correction(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    y_channel = image[:, :, 0]
    y_channel = 255. * (y_channel / 255.) ** (1.3 ** (y_channel.mean() / 128.))
    image[:, :, 0] = y_channel

    image = cv.cvtColor(image, cv.COLOR_YCrCb2BGR)
    return image


def compute_low_high_prop(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7, 7), 15, None, 15)
    count_low = blurred[blurred < 60].flatten().shape[-1]
    count_high = blurred[blurred > 195].flatten().shape[-1]

    total_black_count = blurred[blurred <= 5].flatten().shape[-1]
    total_white_count = blurred[blurred >= 250].flatten().shape[-1]

    low_prop = (count_low - total_black_count) / (blurred.flatten().shape[-1] - total_black_count)
    high_prop = (count_high - total_white_count) / (blurred.flatten().shape[-1] - total_white_count)
    return low_prop, high_prop


def pipeline(image, wb_threshold=0.5, wb_threshold2=0.5):
    image = noise_removal(image)

    low_prop, high_prop = compute_low_high_prop(image)

    if (low_prop > 0.5 and high_prop < 0.1) or (low_prop < 0.1 and high_prop > 0.6):
        image = gamma_correction(image)
        low_prop, high_prop = compute_low_high_prop(image)

    if low_prop + high_prop > 0.6 and (low_prop > 0.4 and high_prop > 0.15):
        image = local_gamma_correction(image)
        image = contrast_stretch(image)

    image = white_balance(image, wb_threshold, wb_threshold2)

    return image
