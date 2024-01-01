import numpy as np
import cv2 as cv
import os


class data_loader:
    def __init__(self, dataframe, directory, batch_size, resize_shape) -> None:
        self.dataframe = dataframe
        self.directory = directory
        self.batch_size = batch_size
        self.resize_shape = resize_shape

    def number_of_batch(self):
        return int(len(self.dataframe) / self.batch_size)

    def shuffle_dataframe(self):
        self.dataframe = self.dataframe.sample(frac=1)

    def get_batch(self, batch_num, preprocessing=None):
        start = batch_num * self.batch_size
        end = start + self.batch_size
        if end > len(self.dataframe):
            end = len(self.dataframe)

        filenames = self.dataframe.iloc[start:end]['filename'].values
        labels = self.dataframe.iloc[start:end]['label'].values
        images = np.zeros((self.batch_size, self.resize_shape[0], self.resize_shape[1], 3))
        for i, filename in enumerate(filenames):
            image = cv.imread(self.directory + filename)
            image = image[:, :, ::-1]
            image = cv.resize(image, self.resize_shape)
            images[i] = image

        if preprocessing is not None:
            images = preprocessing(images)

        return images, labels, filenames


class NoLabelDataLoader:
    def __init__(self, directory, batch_size, target_size):
        self.directory = directory
        self.batch_size = batch_size
        self.target_size = target_size
        self.image_names = os.listdir(directory)
        self.target_size = target_size

    def number_of_batches(self):
        batch_num = len(self.image_names) // self.batch_size
        if len(self.image_names) % self.batch_size != 0:
            batch_num += 1
        return batch_num

    def image_count(self):
        return len(self.image_names)

    def get_batch(self, batch_num, preprocessing=None):
        start = batch_num * self.batch_size
        end = start + self.batch_size
        if end > len(self.image_names):
            end = len(self.image_names)

        filenames = self.image_names[start:end]
        images = np.zeros((len(filenames), self.target_size[0], self.target_size[1], 3))

        for i, filename in enumerate(filenames):
            image = cv.imread(self.directory + filename)
            image = image[:, :, ::-1]
            image = cv.resize(image, self.target_size)
            images[i] = image

        if preprocessing is not None:
            images = preprocessing(images)

        return images, filenames
