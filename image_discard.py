import numpy as np
import pandas as pd
import misc


def find_outliers_per_class_centered_scaled(standardized_features, labels, image_names, threshold=0.2):
    classes = np.unique(labels)
    distances = [[] for _ in range(len(classes))]
    names = [[] for _ in range(len(classes))]
    for c in classes:
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
    cleaned_df.to_csv(base_dir + filename + '.csv', header=False)
    return cleaned_df
