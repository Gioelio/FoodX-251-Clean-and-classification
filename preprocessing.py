import cv2 as cv
import numpy as np


def noise_removal(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)

    ksize = 15

    image[:, :, 1] = cv.GaussianBlur(image[:, :, 1], (ksize, ksize), 9, None, 9)
    image[:, :, 2] = cv.GaussianBlur(image[:, :, 2], (ksize, ksize), 9, None, 9)

    image = cv.cvtColor(image, cv.COLOR_YCrCb2BGR)
    image = cv.bilateralFilter(image, 11, 150, 15)
    image = cv.GaussianBlur(image, (3, 3), 5, None, 5)
    return image


def white_balance(image, threshold1=0.5, threshold2=0.5):
    gray = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)[:, :, 0]
    sat = (np.max(image, axis=2) - np.min(image, axis=2)) / np.max(image, axis=2)
    sat[np.isnan(sat)] = 0

    if sat.mean() > 0.5:
        sat_threshold = np.quantile(sat.flatten(), threshold1)
        wb = cv.xphoto.createSimpleWB()

        wb.setP(sat_threshold)
        image = wb.balanceWhite(image)

    else:
        sat_threshold = np.quantile(sat.flatten(), (1 - threshold2))
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
    # cv.imshow('mask', mask)
    # cv.waitKey(0)
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


def gamma_correction(image, type):
    y = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    q1 = np.quantile(y[:, :, 0], 0.5)
    if type == 'low':
        print(q1)
        y[:, :, 0] = (y[:, :, 0] / 255.) ** 0.7 * 255
    else:
        y[:, :, 0] = (y[:, :, 0] / 255.) ** 1.3 * 255

    image = cv.cvtColor(y, cv.COLOR_YCrCb2BGR)
    return image


def pipeline(image, wb_threshold=0.5, wb_threshold2=0.5):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gradient = cv.Laplacian(gray, ddepth=3)
    mean, std = cv.meanStdDev(gradient)
    image = white_balance(image, wb_threshold, wb_threshold2)

    blurred = cv.GaussianBlur(gray, (7, 7), 15, None, 15)
    count_low = blurred[blurred < 75].flatten().shape[-1]
    count_high = blurred[blurred > 180].flatten().shape[-1]

    total_black_count = blurred[blurred <= 5].flatten().shape[-1]
    total_white_count = blurred[blurred >= 250].flatten().shape[-1]

    low_prop = (count_low - total_black_count) / (blurred.flatten().shape[-1] - total_black_count)
    high_prop = (count_high - total_white_count) / (blurred.flatten().shape[-1] - total_white_count)

    if low_prop + high_prop > 0.7 and (low_prop > 0.5 and high_prop > 0.2):
        # cv.destroyAllWindows()
        # cv.imshow('before', image)
        # cv.waitKey(1)
        image = local_gamma_correction(image)
        image = contrast_stretch(image, 2)
        # cv.imshow(str(low_prop) + " " + str(high_prop), image)
        # cv.waitKey(0)

    elif low_prop > 0.7 or high_prop > 0.7:
        # cv.destroyAllWindows()
        # cv.imshow('before', image)
        # cv.waitKey(1)
        image = gamma_correction(image, 'low' if low_prop > 0.7 else 'high')
        # cv.imshow(str(low_prop) + " " + str(high_prop), image)
        # cv.waitKey(0)
        pass

    #
    if std[0, 0] > 100:
        image = noise_removal(image)
        # contrast_stretch(image, 5)

    return image
