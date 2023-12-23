import cv2 as cv
import numpy as np


def sift(image, max_features=None):
    detector = cv.SIFT.create(nfeatures=max_features)
    _, features = detector.detectAndCompute(cv.cvtColor(image, cv.COLOR_BGR2GRAY), None)
    return features


def gabor_response(image, size, angles, sigmas, lambdas, gammas):
    import itertools
    final_vector = []
    for angle, sig, lm, gm in itertools.product(*[angles, sigmas, lambdas, gammas]):
        filter2d = cv.getGaborKernel(size, sig, angle, lm, gm)
        response = np.array(
            cv.filter2D(src=cv.cvtColor(image, cv.COLOR_BGR2GRAY), ddepth=-1, kernel=filter2d)).flatten()
        final_vector.append(np.mean(response))

    return np.array(final_vector)


def color_histograms(image, bins=20, flatten=False):
    y_cb_cr = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    channels = cv.split(y_cb_cr)
    histogram = np.zeros(shape=(3, bins))
    for i, channel in enumerate(channels):
        histogram[i] = cv.calcHist([channel], [0], None,
                                   [bins], [0, 256]).reshape((bins,))
    if flatten:
        return histogram.flatten()
    return histogram


def extract_features(directory, image_names, callbacks, image_size=None):
    import tqdm
    features = []
    for i, name in tqdm.tqdm(enumerate(image_names)):
        features_array = None
        image = cv.imread(directory + name)
        if image_size is not None:
            image = cv.resize(image, image_size)
        for callback in callbacks:
            f = callback(image)
            if features_array is not None:
                features_array = np.concatenate((features_array, f))
            else:
                features_array = f

        features.append(features_array)

    return features
