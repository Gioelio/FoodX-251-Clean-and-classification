import numpy as np
import cv2 as cv

class data_loader:
    def __init__(self, dataframe, directory, batch_size, resize_shape) -> None:
        self.dataframe = dataframe
        self.directory = directory
        self.batch_size = batch_size
        self.resize_shape = resize_shape;

    def number_of_batch(self):
        return int(len(self.dataframe) / self.batch_size);

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
            if preprocessing is not None:
                image = preprocessing(image)
            images[i] = image

        return images, labels, filenames