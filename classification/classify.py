import numpy as np
from tensorflow import keras
import cv2 as cv
from vit_keras import vit
from keras.applications.efficientnet_v2 import preprocess_input
import gc
from scipy.special import softmax
from scipy.stats import entropy
from misc import load_class_labels

PRETRAINED_VIT_PATH = "classification/tuned_models/ViTb16_noise_extended"
PRETRAINED_EFFICIENTNET_PATH = "classification/tuned_models/efficientnet_v2_noise_extended"
LABEL_NAMES_PATH = "dataset/classes.txt"


def load_and_predict(model_path, image, preprocessing):
    trained_model = keras.models.load_model(model_path)
    model_input = preprocessing(image)
    prediction = trained_model.predict(model_input)[0]
    del trained_model
    keras.backend.clear_session()
    gc.collect()
    return prediction


def classify_image(path):
    classnames = load_class_labels('dataset/classes.txt')
    image = cv.imread(path)[:, :, ::-1]
    image = cv.resize(image, (224, 224))
    image = np.expand_dims(image, 0)

    vit_prediction = load_and_predict(PRETRAINED_VIT_PATH, image, vit.preprocess_inputs)
    efficientnet_prediction = load_and_predict(PRETRAINED_EFFICIENTNET_PATH, image, preprocess_input)

    vit_entropy = 1 / entropy(vit_prediction)
    en_entropy = 1 / entropy(efficientnet_prediction)

    normalized_vit_entropy = vit_entropy / (vit_entropy + en_entropy)
    normalized_en_entropy = en_entropy / (en_entropy + vit_entropy)

    ensemble_prediction = normalized_vit_entropy * vit_prediction + normalized_en_entropy * efficientnet_prediction
    ordered_classes = np.argsort(-ensemble_prediction)

    return ordered_classes, ensemble_prediction, classnames
