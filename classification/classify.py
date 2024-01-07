import numpy as np
from tensorflow import keras
import cv2 as cv
from vit_keras import vit
from keras.applications.efficientnet_v2 import preprocess_input
from scipy.stats import entropy
from misc import load_class_labels


PRETRAINED_VIT_PATH = "classification/tuned_models/ViTb16_noise_extended"
PRETRAINED_EFFICIENTNET_PATH = "classification/tuned_models/efficientnet_v2_noise_extended"
LABEL_NAMES_PATH = "dataset/classes.txt"

EN_MODEL = keras.models.load_model(PRETRAINED_EFFICIENTNET_PATH)
VIT_MODEL = keras.models.load_model(PRETRAINED_VIT_PATH)


def classify_image(path):
    classnames = load_class_labels('dataset/classes.txt')
    image = cv.imread(path)[:, :, ::-1]
    image = cv.resize(image, (224, 224))
    image = np.expand_dims(image, 0)

    vit_prediction = VIT_MODEL.predict(vit.preprocess_inputs(image))[0]
    efficientnet_prediction = EN_MODEL.predict(preprocess_input(image))[0]

    vit_entropy = 1 / entropy(vit_prediction)
    en_entropy = 1 / entropy(efficientnet_prediction)

    normalized_vit_entropy = vit_entropy / (vit_entropy + en_entropy)
    normalized_en_entropy = en_entropy / (en_entropy + vit_entropy)

    ensemble_prediction = normalized_vit_entropy * vit_prediction + normalized_en_entropy * efficientnet_prediction
    ordered_classes = np.argsort(-ensemble_prediction)

    return ordered_classes, ensemble_prediction, classnames
