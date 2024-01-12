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
from feature_extraction import build_feature_extractor
from vit_keras import vit
from misc import import_model, import_general

HANDCRAFTED_FEATURES = 'dataset/handcrafted/'
EN_FEATURES = 'similarity_search/extracted_features/efficient_net_extended_similarity'
EN_FILENAMES = 'similarity_search/extracted_features/efficient_net_extended_similarity_filenames.csv'

BASE_FEATURES = 'similarity_search/extracted_features/efficient_net_not_tuned_similarity'
BASE_FILENAMES = 'similarity_search/extracted_features/efficient_net_not_tuned_similarity_filenames.csv'

VIT_FEATURES = 'similarity_search/extracted_features/vitb16_extended_similarity'
VIT_FILENAMES = 'similarity_search/extracted_features/vitb16_extended_similarity_filenames.csv'

IMAGE_DIR = 'dataset/complete/'

VIT_FEATURE_EXTRACTOR = import_model('classification/tuned_models/ViTb16_noise_extended')
VIT_FEATURE_EXTRACTOR = keras.Sequential(VIT_FEATURE_EXTRACTOR.layers[:-1])

EN_FEATURE_EXTRACTOR = import_model('classification/tuned_models/efficientnet_v2_noise_extended')
EN_FEATURE_EXTRACTOR = keras.Sequential(EN_FEATURE_EXTRACTOR.layers[:-1])

BASE_FEATURES_EXTRACTOR = build_feature_extractor(EfficientNetV2B0, 'top_dropout')


def load_images_features(load_handcrafted=False):
    vit_features = import_general(VIT_FEATURES + '.npy', lambda x: np.load(x))
    en_features = import_general(EN_FEATURES + '.npy', lambda x: np.load(x))
    base_features = import_general(BASE_FEATURES + '.npy', lambda x: np.load(x))
    handcrafted_features = None
    if load_handcrafted:
        handcrafted_features, _ = import_general(HANDCRAFTED_FEATURES, lambda x: load_all_features(x, import_general(
            IMAGE_DIR, lambda x: os.listdir(x)),
                                                                                                   load_color=True,
                                                                                                   load_lbp=True,
                                                                                                   load_gabor=False,
                                                                                                   load_sift=False,
                                                                                                   load_bow=False))

    return (vit_features, en_features, base_features), handcrafted_features


def find_similar_distances(query_path, features_nn, base_weight=None, features_handcrafted=None):
    image_limit = 10000
    import_filenames = lambda path: import_general(path, lambda x: pd.read_csv(x, header=None).iloc[:, 1].values)

    base_most_similar = None
    vit_most_similar = None
    en_most_similar = None
    handcrafted_most_similar = None
    
    vit_distances = None
    en_distances = None
    base_distances = None
    handcrafted_distances = None

    intersection = None

    if base_weight is None or base_weight != 1:
        vit_most_similar, vit_distances = find_similar(VIT_FEATURE_EXTRACTOR, query_path, features_nn[0],
                                                       import_filenames(VIT_FILENAMES),
                                                       vit.preprocess_inputs, output_number=image_limit)
        en_most_similar, en_distances = find_similar(EN_FEATURE_EXTRACTOR, query_path, features_nn[1],
                                                     import_filenames(EN_FILENAMES),
                                                     preprocess_input, output_number=image_limit)
        intersection = np.intersect1d(vit_most_similar, en_most_similar)

    if base_weight is None or base_weight != 0:
        base_most_similar, base_distances = find_similar(BASE_FEATURES_EXTRACTOR, query_path, features_nn[2],
                                                         import_filenames(BASE_FILENAMES),
                                                         preprocess_input, output_number=image_limit)
        if intersection is not None:
            intersection = np.intersect1d(intersection, base_most_similar)
        else:
            intersection = base_most_similar

    if features_handcrafted is not None:
        handcrafted_most_similar, handcrafted_distances = find_similar_handcrafted(IMAGE_DIR, features_handcrafted, query_path,
                                                                       False, output_number=image_limit)

        intersection = np.intersect1d(handcrafted_most_similar, nn_most_similar)

    return (vit_most_similar, vit_distances), (en_most_similar, en_distances), (base_most_similar, base_distances), (handcrafted_most_similar, handcrafted_distances), intersection


def weight_filenames(vit_info, en_info, base_info, hand_info, intersection, base_weight, use_nn=True, use_intersection=False, verbose=1):
    (vit_most_similar, vit_distances) = vit_info;
    (en_most_similar, en_distances) = en_info;
    (base_most_similar, base_distances) = base_info;
    (handcrafted_most_similar, handcrafted_distances) = hand_info;

    sums = {}
    for el in intersection:
        sums[el] = 0

        if vit_most_similar is not None:
            i = np.where(vit_most_similar == el)
            sums[el] += vit_distances[i] * (0.5 if base_weight == 0 else base_weight / 2)

            j = np.where(en_most_similar == el)
            sums[el] += en_distances[j] * (0.5 if base_weight == 0 else base_weight / 2)

        if base_most_similar is not None:
            k = np.where(base_most_similar == el)
            sums[el] += base_distances[k] * ((1 - base_weight) if base_weight < 1 else 1)

    nn_most_similar = {k: v for k, v in sorted(sums.items(), key=lambda item: item[1])}

    most_similar_filenames = nn_most_similar

    most_similar_filenames = nn_most_similar if use_nn else handcrafted_most_similar
    most_similar_filenames = intersection if use_intersection else most_similar_filenames

    most_similar_filenames = [IMAGE_DIR + filename for filename in most_similar_filenames]

    if verbose:
        print('end')

    return most_similar_filenames

## deprecated
def find_similar_images(query_path, features_handcrafted, features_nn, use_intersection, use_nn, base_weight=0.5,
                        verbose=1):

    vit_info, en_info, base_info, intersection = find_similar_distances(query_path, features_nn, verbose)
    return weight_filenames(vit_info, en_info, base_info, intersection, base_weight, use_nn, use_intersection, verbose)
    
