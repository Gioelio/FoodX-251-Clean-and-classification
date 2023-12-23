import numpy as np


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


def write_discarded_images(names, class_labels, discarded_dir, images_dir):
    import os
    import shutil
    for c in range(len(names)):
        discarded_subdir = discarded_dir + str(c) + "_" + class_labels[c]
        os.makedirs(discarded_subdir)
        for name in names[c]:
            shutil.copyfile(images_dir + name, discarded_subdir + '/' + name)
