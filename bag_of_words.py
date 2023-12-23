from sklearn.cluster import KMeans
import numpy as np


def fit(features, vocabulary_size=200, max_iter=50, n_init=5, verbose=False):
    kmeans = KMeans(n_clusters=vocabulary_size, verbose=verbose, max_iter=max_iter, n_init=n_init)
    return kmeans.fit(features)


def predict(kmeans, features):
    bag_of_words = np.zeros((len(features), len(kmeans.cluster_centers_)))
    for i, image in enumerate(features):
        if image is not None:
            word = kmeans.predict(image)
            for w in word:
                bag_of_words[i, w] += (1 / image.shape[0])
    return bag_of_words
