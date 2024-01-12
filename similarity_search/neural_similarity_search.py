from tqdm import tqdm
from data_loader import NoLabelDataLoader
from tensorflow import keras
import gc
import numpy as np
import cv2 as cv
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist


def extract_features(model_builder, path, rebuild_interval, preprocessing, batch_size, image_size):
    loader = NoLabelDataLoader(path, batch_size=batch_size, target_size=image_size)
    feature_extractor = model_builder()
    extracted_features = np.zeros((loader.image_count(), feature_extractor.output_shape[-1]))
    filenames = []

    for i in tqdm(range(loader.number_of_batches())):
        if i > 0 and i % rebuild_interval == 0:
            del feature_extractor
            keras.backend.clear_session()
            gc.collect()
            feature_extractor = model_builder()
        images, names = loader.get_batch(i, preprocessing=preprocessing)
        extracted_features[i * batch_size: i * batch_size + len(images), :] = feature_extractor.predict(images,
                                                                                                        verbose=False,
                                                                                                        batch_size=16)
        for f in names:
            filenames.append(f)

    return extracted_features, filenames


def find_similar(feature_extractor, query_image_path, database_features, database_names, preprocessing, norm=True,
                 output_number=10):
    if norm:
        database_features = normalize(database_features)
    query_image = cv.imread(query_image_path)[:, :, ::-1]
    query_image = cv.resize(query_image, (feature_extractor.input_shape[1], feature_extractor.input_shape[2]))
    query_image = preprocessing(query_image)
    img_array = np.zeros((1, feature_extractor.input_shape[1], feature_extractor.input_shape[2], 3))
    img_array[0] = query_image
    extracted_query = feature_extractor.predict(img_array, verbose=0)
    if norm:
        extracted_query = normalize(extracted_query)

    distances = cdist(extracted_query, database_features, 'cosine')[0]
    most_similar_indices = np.argsort(distances)[0:output_number]

    return database_names[most_similar_indices], distances[most_similar_indices]

def find_similar_handcrafted(images_dir, features, query_path, norm=True, output_number=10):
    import pandas as pd
    import os

    if norm:
        features = features(normalize);

    arr = []
    for feat in features:
        arr.append(np.array(feat))

    filenames = os.listdir(images_dir)

    df = pd.DataFrame({'features': arr, 'filenames': filenames})
    
    mask = [el in query_path for el in df['filenames']]
    query_features = df[mask].iloc[0].values[0]
    distances = cdist([query_features], features)[0]
    most_similar = np.argsort(distances)[0:output_number]

    return df['filenames'][most_similar].values, distances[most_similar]


def order_prediction(prediction, significance_threshold=None):
    ordered_classes = np.argsort(-prediction)
    prediction = prediction[ordered_classes]

    if significance_threshold is not None:
        cum_prob = prediction.cumsum()
        cum_prob = [(x, i) for x, i in enumerate(cum_prob)]
        cum_prob = [pair for pair in cum_prob if pair[1] > significance_threshold][0]
        index_limit = cum_prob[0] + 1;
        return prediction[:index_limit], ordered_classes[:index_limit]
    
    return prediction, ordered_classes

def filter_images_not_in_same_class(predictions, model, preprocess_input, query_path, most_similar_filenames, significance_threshold=0.7, limit_number_of_classes=5):
    img = cv.imread(query_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, 0)
    query_pred = model.predict(img, verbose=0)[0]

    pred, ordered_classes = order_prediction(query_pred, significance_threshold)
    pred = pred[:limit_number_of_classes]
    ordered_classes = ordered_classes[:limit_number_of_classes]

    valid = predictions[predictions['filenames'].isin(most_similar_filenames)]

    mask = [len(np.intersect1d(prediction, ordered_classes)) > 0 for prediction in valid['labels']]
    valid = valid[mask]
    valid = valid.sort_values(by='filenames', key=lambda x: x.map({k: i for i, k in enumerate(most_similar_filenames)}))
    return valid['filenames'].values.tolist(), mask;