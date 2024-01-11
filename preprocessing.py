import cv2 as cv
import numpy as np
from brisque import brisque


def noise_removal(image):
    gradient = cv.Laplacian(cv.cvtColor(image, cv.COLOR_BGR2GRAY), ddepth=3)
    mean, std = cv.meanStdDev(gradient)

    if std[0, 0] > 100:
        image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
        image[:, :, 0] = cv.fastNlMeansDenoising(image[:, :, 0], None, 0, 7, 21)
        image[:, :, 1] = cv.fastNlMeansDenoising(image[:, :, 1], None, 20, 7, 21)
        image[:, :, 2] = cv.fastNlMeansDenoising(image[:, :, 2], None, 20, 7, 21)
        image = cv.cvtColor(image, cv.COLOR_YCrCb2BGR)

        image = cv.bilateralFilter(image, 15, 70, 15)
        image = cv.bilateralFilter(image, 7, 70, 15)
        image = cv.GaussianBlur(image, (3, 3), 3, None, 3)
    return image


def white_balance(image, method='simple'):
    if method == 'simple':
        wb = cv.xphoto.createSimpleWB()
        wb.setP(0.01)
    else:
        wb = cv.xphoto.createGrayworldWB()
        wb.setSaturationThreshold(0.4)
    image = wb.balanceWhite(image)
    return image


def local_gamma_correction(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    mask = image[:, :, 0]
    mask = 255 - cv.bilateralFilter(mask, 11, 80, 60)
    y_channel = image[:, :, 0]
    y_channel = 255. * (y_channel / 255.) ** (2. ** ((128. - mask) / 128.))
    image[:, :, 0] = y_channel

    image = cv.cvtColor(image, cv.COLOR_YCrCb2BGR)
    return image


def contrast_stretch(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    channel = 0
    maximum = image[:, :, channel].flatten().max()
    minimum = image[:, :, channel].flatten().min()
    image[:, :, channel] = (image[:, :, channel] - minimum) * float(255. / float(maximum - minimum))
    image = cv.cvtColor(image, cv.COLOR_YCrCb2BGR)
    return image


def pipeline(image, wb='simple'):
    gradient = cv.Laplacian(cv.cvtColor(image, cv.COLOR_BGR2GRAY), ddepth=3)
    mean, std = cv.meanStdDev(gradient)

    if std[0, 0] > 100:
        image = noise_removal(image)

    image = white_balance(image, wb)
    image = contrast_stretch(image)

    if std[0, 0] > 100:
        image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
        image[:, :, 0] = cv.fastNlMeansDenoising(image[:, :, 0], None, 13, 3, 21)
        image = cv.cvtColor(image, cv.COLOR_YCrCb2BGR)

    return image
