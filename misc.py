import numpy as np
import pandas as pd
import os
import shutil


def load(filename, column=1, stratified_sample_rate=1):
    dataframe = pd.read_csv(filename, names=['filename', 'label'], header=None)
    dataframe = dataframe.groupby(dataframe.columns[column], group_keys=False).apply(
        lambda group: group.sample(frac=stratified_sample_rate)
    )
    return dataframe


def load_class_labels(path, column=1, sep=' ', header=None):
    df = pd.read_csv(path, sep=sep, header=header)
    return df.iloc[:, column].values


def group_mean(array, labels):
    _ndx = np.argsort(labels)
    _id, _pos, count = np.unique(labels[_ndx],
                                 return_index=True,
                                 return_counts=True)
    _sum = np.add.reduceat(array[_ndx], _pos, axis=0)
    return _sum / count[:, None]


def is_positive_semidefinite(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")
    return np.all(np.linalg.eigvals(matrix) >= 0)


def group_stats(array, labels, numeric_correction=1e-10):
    mean = np.zeros((len(np.unique(labels)), array.shape[1]))
    cov = np.zeros((len(np.unique(labels)), array.shape[1], array.shape[1]))
    for i, c in enumerate(np.unique(labels)):
        indices = labels == c
        sub = array[indices]
        cov[i] = np.cov(sub, rowvar=False)
        for j in range(cov[i].shape[0]):
            cov[i, j, j] += numeric_correction
        if not is_positive_semidefinite(cov[i]):
            print("Non positive semidefinite matrix")
        mean[i] = sub.mean(axis=0)

    return mean, cov


def unroll_arrays(arrays_list, sampling_frac=1):
    result = []
    for el in arrays_list:
        if el is not None:
            for i in el:
                if np.random.random() < sampling_frac:
                    result.append(i)
    return np.asarray(result)


def center_scale_columns(features, labels):
    res = features.copy()
    for c in np.unique(labels):
        indices = labels == c
        std = res[indices].std(axis=0)
        for col in range(std.shape[0]):
            if std[col] != 0:
                res[indices, col] = (res[indices, col] - res[indices, col].mean(axis=0)) / res[indices, col].std(axis=0)
            else:
                res[indices, col] = (res[indices, col] - res[indices, col].mean(axis=0))
    return res


def apply_data_augmentation(images):
    from imgaug import augmenters as iaa

    seq = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 5))),
        # loc and scale are mean and std of the gaussian noise from to sample
        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(loc=0, scale=(0.1 * 255, 0.5 * 255), per_channel=0.9)),
        iaa.Sometimes(0.5, iaa.JpegCompression(compression=(85, 99)))
    ], random_order=True)

    images = images.astype('uint8')
    images_aug = seq(images=images)

    return images_aug


def apply_standard_data_augmentation(images):
    from imgaug import augmenters as iaa
    seq = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.HorizontalFlip()),
        iaa.Sometimes(0.5, iaa.Rotate()),
        iaa.Sometimes(0.5, iaa.Crop())
    ], random_order=True)
    return seq(images=images)


def create_clean_trainset(train_dir, clean_train_dir, clean_train_names, clean_train_labels, class_names):
    os.mkdir(clean_train_dir)
    for c in np.unique(clean_train_labels):
        class_directory = clean_train_dir + str(c) + '_' + class_names[c] + '/'
        os.mkdir(class_directory)
        names = clean_train_names[clean_train_labels == c]
        for name in names:
            shutil.copy(train_dir + name, class_directory + name)
