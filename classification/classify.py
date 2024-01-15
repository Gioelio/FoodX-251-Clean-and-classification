import numpy as np
import cv2 as cv
from vit_keras import vit
from keras.applications.efficientnet_v2 import preprocess_input
from misc import load_class_labels, import_model
from preprocessing import pipeline
from brisque import brisque

PRETRAINED_VIT_PATH = "classification/tuned_models/ViTb16_noise_extended"
PRETRAINED_EFFICIENTNET_PATH = "classification/tuned_models/efficientnet_v2_noise_extended"
LABEL_NAMES_PATH = "dataset/classes.txt"

EN_MODEL = import_model(PRETRAINED_EFFICIENTNET_PATH)
VIT_MODEL = import_model(PRETRAINED_VIT_PATH)


def classify_image(path):
    classnames = load_class_labels('dataset/classes.txt')
    image = cv.imread(path)

    image = pipeline(image)
    q = brisque.BRISQUE()
    quality = q.score(image)
    if quality >= 85:
        raise Exception("The given input image has a quality score of {0:.2f}. That's too high!".format(quality))
    image = image[:, :, ::-1]
    image = cv.resize(image, (224, 224))
    image = np.expand_dims(image, 0)

    vit_prediction = VIT_MODEL.predict(vit.preprocess_inputs(image))[0]
    en_prediction = EN_MODEL.predict(preprocess_input(image))[0]

    ensemble_prediction = np.sqrt(vit_prediction * en_prediction)
    ensemble_prediction /= ensemble_prediction.sum()
    ordered_classes = np.argsort(-ensemble_prediction)

    return ordered_classes, ensemble_prediction, classnames
