import numpy as np
from tqdm import tqdm_notebook
import misc
import cv2 as cv


def find_outliers_per_class_centered_scaled(standardized_features, labels, image_names, threshold=0.2):
    classes = np.unique(labels)
    distances = [[] for _ in range(len(classes))]
    names = [[] for _ in range(len(classes))]
    for c in tqdm_notebook(classes):
        indices = labels == c
        for image, name in zip(standardized_features[indices], image_names[indices]):
            distance = np.linalg.norm(image)
            if distance >= threshold:
                distances[c].append(distance)
                names[c].append(name)

        distances[c] = np.asarray(distances[c])
        names[c] = np.asarray(names[c])
        _sorted = np.argsort(distances[c])[::-1]
        distances[c] = distances[c][_sorted]
        names[c] = names[c][_sorted]

    return names, distances


def find_outliers_iter(features, labels, image_names, threshold=15, num_iter=10):
    to_consider = np.ones((len(features)))
    classes = np.unique(labels)
    distances = [[] for _ in range(len(classes))]
    names = [[] for _ in range(len(classes))]

    for _ in range(num_iter):
        features_to_consider = features[to_consider == 1]
        names_to_consider = image_names[to_consider == 1]
        labels_to_consider = labels[to_consider == 1]
        standardized_features = misc.center_scale_columns(features_to_consider, labels_to_consider)
        n, d = find_outliers_per_class_centered_scaled(standardized_features, labels_to_consider, names_to_consider,
                                                       threshold)
        for i in range(len(n)):
            for j in range(len(d[i])):
                names[i].append(n[i][j])
                distances[i].append(d[i][j])
                to_consider[np.where(image_names == n[i][j])[0]] = 0

    return names, distances


def find_outliers_per_class(bow_mean, bow_cov, bow, labels, image_names, threshold=13):
    from scipy.spatial.distance import mahalanobis
    classes = np.unique(labels)
    distances = [[] for _ in range(len(classes))]
    names = [[] for _ in range(len(classes))]

    for c in classes:
        inv = np.linalg.inv(bow_cov[c])
        class_mean = bow_mean[c]
        indices = labels == c
        for image, name in zip(bow[indices], image_names[indices]):
            distance = mahalanobis(image, class_mean, inv)
            if distance >= threshold:
                distances[c].append(distance)
                names[c].append(name)

        distances[c] = np.asarray(distances[c])
        names[c] = np.asarray(names[c])
        _sorted = np.argsort(distances[c])[::-1]
        distances[c] = distances[c][_sorted]
        names[c] = names[c][_sorted]

    return names, distances


def find_outliers_median(features, labels, image_names, threshold=30):
    from sklearn.preprocessing import normalize
    classes = np.unique(labels)
    distances = [[] for _ in range(len(classes))]
    names = [[] for _ in range(len(classes))]
    for c in classes:
        indices = labels == c
        to_consider = features[indices]
        normalized = normalize(to_consider, axis=1)
        median = np.median(normalized, axis=0)

        for image, name in zip(normalized, image_names[indices]):
            d1 = np.linalg.norm((image - median))

            if d1 >= threshold:
                distances[c].append(d1)
                names[c].append(name)

        distances[c] = np.asarray(distances[c])
        names[c] = np.asarray(names[c])
        _sorted = np.argsort(distances[c])[::-1]
        distances[c] = distances[c][_sorted]
        names[c] = names[c][_sorted]
    return names, distances


def find_outliers_iter_median(features, labels, image_names, threshold=30, iter=10):
    from tqdm.notebook import tqdm_notebook
    to_consider = np.ones((len(features)))
    classes = np.unique(labels)
    distances = [[] for _ in range(len(classes))]
    names = [[] for _ in range(len(classes))]

    for _ in tqdm_notebook(range(iter)):
        features_to_consider = features[to_consider == 1]
        names_to_consider = image_names[to_consider == 1]
        labels_to_consider = labels[to_consider == 1]

        n, d = find_outliers_median(features_to_consider, labels_to_consider, names_to_consider,
                                    threshold)
        for i in range(len(n)):
            for j in range(len(d[i])):
                names[i].append(n[i][j])
                distances[i].append(d[i][j])
                to_consider[np.where(image_names == n[i][j])[0]] = 0

    return names, distances

def write_discarded_images(names, class_labels, discarded_dir, images_dir, delete_old=True):
    import os
    import shutil
    if delete_old and os.path.exists(discarded_dir):
        shutil.rmtree(discarded_dir)
    for c in range(len(names)):
        discarded_subdir = discarded_dir + str(c) + "_" + class_labels[c]
        os.makedirs(discarded_subdir)
        for name in names[c]:
            shutil.copyfile(images_dir + name, discarded_subdir + '/' + name)


def write_cleaned_csv(train_df, exclude_names, base_dir, filename='train_info_cleaned'):
    cleaned_df = train_df.copy(deep=True)
    global_exclude = []
    for c in exclude_names:
        for name in c:
            global_exclude.append(name)
    cleaned_df = cleaned_df[~cleaned_df['filename'].isin(global_exclude)]
    cleaned_df.to_csv(base_dir + filename + '.csv', header=False, index=False)
    return cleaned_df


def discard_preprocessing(dir, image_names, labels, bw_threshold=0.9, no_gradient_threshold=0.83, zero_threshold=0.6,
                          gradient_threshold=0.15, crop_size=0.6):

    discarded_images = [[] for _ in range(251)]
    discarded_indices = []

    for i, (image_name, c) in tqdm_notebook(enumerate(zip(image_names, labels))):
        color_image = cv.imread(dir + image_name)
        image = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
        flattened = image.flatten()
        flattened_mask = ((flattened <= 30) | (flattened >= 220))
        bw_proportion = len(flattened[flattened_mask]) / len(flattened)

        if bw_proportion > bw_threshold:
            discarded_images[c].append(image_name)
            discarded_indices.append(i)
            continue

        center = image.shape
        w = center[1] * crop_size
        h = center[0] * crop_size
        x = center[1] / 2 - w / 2
        y = center[0] / 2 - h / 2

        cropped = image[int(y):int(y + h), int(x):int(x + w)]
        gradient = cv.Laplacian(cropped, 0).flatten()
        no_gradient_rate = len(gradient[abs(gradient) < 1].flatten()) / len(gradient)
        if no_gradient_rate >= no_gradient_threshold:
            discarded_images[c].append(image_name)
            discarded_indices.append(i)
            continue

        gradient_rate = len(gradient[abs(gradient) >= 130].flatten()) / len(gradient[abs(gradient) > 0].flatten())
        zero_rate = len(gradient[(abs(gradient) <= 0)].flatten()) / len(gradient.flatten())

        if gradient_rate >= gradient_threshold and zero_rate >= zero_threshold:
            discarded_images[c].append(image_name)
            discarded_indices.append(i)

    return discarded_images, discarded_indices
