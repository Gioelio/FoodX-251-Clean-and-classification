import cv2 as cv
import numpy as np

def noise_removal(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    image[:, :, 1] = cv.fastNlMeansDenoising(image[:, :, 1], None, 25, 7, 21)
    image[:, :, 2] = cv.fastNlMeansDenoising(image[:, :, 2], None, 25, 7, 21)
    image = cv.cvtColor(image, cv.COLOR_YCrCb2BGR)
    image = cv.bilateralFilter(image, 9, 130, 15)
    image = cv.GaussianBlur(image, (3, 3), 10, None, 10)
    return image


def white_balance(image, method='simple', threshold=0.5):
    if method == 'simple':
        wb = cv.xphoto.createSimpleWB()
        wb.setP(0.03)
        image = wb.balanceWhite(image)
    else:
        gray = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)[:, :, 0]
        old_min = gray.min()
        old_max = gray.max()
        sat = (np.max(image, axis=2) - np.min(image, axis=2)) / np.max(image, axis=2)
        sat[np.isnan(sat)] = 255
        sat_threshold = np.percentile(sat.flatten(), (1 - threshold) * 100)

        wb = cv.xphoto.createGrayworldWB()
        wb.setSaturationThreshold(sat_threshold)
        image = wb.balanceWhite(image)
        image = contrast_stretch(image, 0, old_min, old_max)

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


def pipeline(image, wb='simple', wb_threshold=0.5):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gradient = cv.Laplacian(gray, ddepth=3)
    mean, std = cv.meanStdDev(gradient)

    blurred = cv.GaussianBlur(gray, (7, 7), 15, None, 15)
    count_low = blurred[blurred < 75].flatten().shape[-1]
    count_high = blurred[blurred > 180].flatten().shape[-1]

    count_black = blurred[blurred < 20].flatten().shape[-1]
    count_white = blurred[blurred > 230].flatten().shape[-1]

    low_prop = (count_low - count_black) / (blurred.flatten().shape[-1] - count_black)
    high_prop = (count_high - count_white) / (blurred.flatten().shape[-1] - count_white)

    if low_prop > 0.4 or high_prop > 0.4:
        # cv.destroyAllWindows()
        # cv.imshow('before', image)
        # cv.waitKey(1)
        image = local_gamma_correction(image)
        image = contrast_stretch(image, 1)
        # cv.imshow(str(low_prop) + " " + str(high_prop), image)
        # cv.waitKey(0)

    image = white_balance(image, wb, wb_threshold)
    if std[0, 0] > 100:
        image = noise_removal(image)
        #contrast_stretch(image, 5)


    return image
