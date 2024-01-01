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
    extracted_query = feature_extractor.predict(img_array)
    if norm:
        extracted_query = normalize(extracted_query)

    distances = cdist(extracted_query, database_features)[0]
    most_similar_indices = np.argsort(distances)[0:output_number]

    return database_names[most_similar_indices], distances[most_similar_indices]
