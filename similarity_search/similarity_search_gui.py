import pandas as pd
import cv2 as cv
import sys
import os
import numpy as np
from tensorflow import keras

sys.path.append('..')

from keras.applications.efficientnet_v2 import EfficientNetV2B0, preprocess_input
from similarity_search.neural_similarity_search import find_similar, find_similar_handcrafted
from handcrafted_extraction import load_all_features

HANDCRAFTED_FEATURES = 'dataset/handcrafted/'
NN_FEATURES = 'similarity_search/extracted_features/efficient_net_tuned_similarity'
NN_FILENAMES = 'similarity_search/extracted_features/efficient_net_tuned_similarity_filenames.csv'
IMAGE_DIR = 'dataset/complete/'


def load_images_for_gui(type='nn'):
    if type == 'nn':
        features = np.load(NN_FEATURES + '.npy')
    else:
        features, _ = load_all_features(HANDCRAFTED_FEATURES, os.listdir(IMAGE_DIR), load_color=True, load_lbp=True, load_gabor=False, load_sift=False, load_bow=False)

    filenames = pd.read_csv(NN_FILENAMES, header=None).iloc[:, 1].values

    return features, filenames

def find_images_from_gui(query_path, features_handcrafted, features_nn, filenames, use_intersection, use_nn):
    from feature_extraction import build_feature_extractor
    img = cv.imread(query_path);
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB);

    image_limit = 1000
    model = keras.models.load_model('classification/tuned_models/efficientnet_v2_cosine')
    model = keras.Sequential(model.layers[:-1])
    nn_most_similar, distances = find_similar(model, query_path, features_nn, filenames, preprocess_input, output_number=image_limit)
    intersection = [];
    if features_handcrafted is not None:
        handcrafted_most_similar, distances = find_similar_handcrafted(IMAGE_DIR, features_handcrafted, query_path, False, output_number=image_limit)
        intersection = np.intersect1d(handcrafted_most_similar, nn_most_similar)

    most_similar_filenames = nn_most_similar if use_nn else handcrafted_most_similar
    most_similar_filenames = intersection if use_intersection else most_similar_filenames

    most_similar_filenames = nn_most_similar
    most_similar_filenames = [IMAGE_DIR + filename for filename in most_similar_filenames];

    return most_similar_filenames;
