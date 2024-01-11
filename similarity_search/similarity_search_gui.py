import pandas as pd
import cv2 as cv
import sys
import os
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import normalize

sys.path.append('..')

from keras.applications.efficientnet_v2 import EfficientNetV2B0, preprocess_input
from similarity_search.neural_similarity_search import find_similar, find_similar_handcrafted
from handcrafted_extraction import load_all_features
from feature_extraction import build_feature_extractor 
from vit_keras import vit

HANDCRAFTED_FEATURES = 'dataset/handcrafted/'
EN_FEATURES = 'similarity_search/extracted_features/efficient_net_extended_similarity'
EN_FILENAMES = 'similarity_search/extracted_features/efficient_net_extended_similarity_filenames.csv'

BASE_FEATURES = 'similarity_search/extracted_features/efficient_net_not_tuned_similarity'
BASE_FILENAMES = 'similarity_search/extracted_features/efficient_net_not_tuned_similarity_filenames.csv'

VIT_FEATURES = 'similarity_search/extracted_features/vitb16_extended_similarity'
VIT_FILENAMES = 'similarity_search/extracted_features/vitb16_extended_similarity_filenames.csv'

IMAGE_DIR = 'dataset/complete/'

VIT_FEATURE_EXTRACTOR = keras.models.load_model('classification/tuned_models/ViTb16_noise_extended')
VIT_FEATURE_EXTRACTOR = keras.Sequential(VIT_FEATURE_EXTRACTOR.layers[:-1])

EN_FEATURE_EXTRACTOR = keras.models.load_model('classification/tuned_models/efficientnet_v2_noise_extended')
EN_FEATURE_EXTRACTOR = keras.Sequential(EN_FEATURE_EXTRACTOR.layers[:-1])

BASE_FEATURES_EXTRACTOR = build_feature_extractor(EfficientNetV2B0, 'block6h_se_reduce')


def load_images_for_gui(type='nn'):
    if type == 'nn':
        vit_features = np.load(VIT_FEATURES + '.npy')
        en_features = np.load(EN_FEATURES + '.npy')
        base_features = np.load(BASE_FEATURES + '.npy')
    else:
        features, _ = load_all_features(HANDCRAFTED_FEATURES, os.listdir(IMAGE_DIR), load_color=True, load_lbp=True,
                                        load_gabor=False, load_sift=False, load_bow=False)

    filenames = pd.read_csv(VIT_FILENAMES, header=None).iloc[:, 1].values

    return (vit_features, en_features, base_features), filenames


def find_images_from_gui(query_path, features_handcrafted, features_nn, filenames, use_intersection, use_nn, base_weight=0.5):
    img = cv.imread(query_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    print('base_weight: ', base_weight)

    image_limit = 10000
    vit_most_similar, vit_distances = find_similar(VIT_FEATURE_EXTRACTOR, query_path, features_nn[0], filenames,
                                                   vit.preprocess_inputs, output_number=image_limit)
    en_most_similar, en_distances = find_similar(EN_FEATURE_EXTRACTOR, query_path, features_nn[1], filenames,
                                                   preprocess_input, output_number=image_limit)
    
    base_filenames = pd.read_csv(BASE_FILENAMES, header=None).iloc[:, 1].values

    base_most_similar, base_distances = find_similar(BASE_FEATURES_EXTRACTOR, query_path, features_nn[2], base_filenames,
                                                     preprocess_input, output_number=image_limit)

    intersection = np.intersect1d(vit_most_similar, en_most_similar)
    #intersection = np.intersect1d(intersection, base_most_similar)

    vit_distances = vit_distances / np.linalg.norm(vit_distances)
    en_distances = en_distances / np.linalg.norm(en_distances)
    base_distances = base_distances / np.linalg.norm(base_distances)

    sums = {}
    for el in intersection:
        i = np.where(vit_most_similar == el)
        sums[el] = vit_distances[i] * (1 - base_weight) / 2
        j = np.where(en_most_similar == el)
        sums[el] += en_distances[j] * (1 - base_weight) / 2
    for el in base_most_similar:
        k = np.where(base_most_similar == el)
        value = base_distances[k] * base_weight;
        if el in sums.keys():
            sums[el] += value
        else:
            sums[el] = value

    nn_most_similar = {k: v for k, v in sorted(sums.items(), key=lambda item: item[1])}

    # i = 0
    # j = 0
    # while i + j < image_limit:
    #     # while vit_most_similar[i] in nn_most_similar:
    #     #     i += 1
    #     # while en_most_similar[j] in nn_most_similar:
    #     #     j += 1
    #     # if vit_distances[i] < en_distances[j]:
    #     if False:
    #         nn_most_similar.append(vit_most_similar[i])
    #         i += 1
    #     else:
    #         nn_most_similar.append(en_most_similar[j])
    #         j += 1


    if features_handcrafted is not None:
        handcrafted_most_similar, distances = find_similar_handcrafted(IMAGE_DIR, features_handcrafted, query_path,
                                                                       False, output_number=image_limit)

        intersection = np.intersect1d(handcrafted_most_similar, nn_most_similar)

    most_similar_filenames = nn_most_similar if use_nn else handcrafted_most_similar
    most_similar_filenames = intersection if use_intersection else most_similar_filenames

    most_similar_filenames = nn_most_similar
    most_similar_filenames = [IMAGE_DIR + filename for filename in most_similar_filenames]

    return most_similar_filenames
