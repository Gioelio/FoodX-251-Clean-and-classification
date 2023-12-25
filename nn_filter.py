import cv2 as cv
import numpy as np
import tensorflow
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from tensorflow import keras
from tqdm import tqdm
from data_loader import data_loader


class NN_filter:
    def __init__(self, train, train_dir, model_name, batch_size=200):
        self.train = train
        self.train_dir = train_dir
        self.batch_size = batch_size
        self.model_name = model_name
        self.dl = data_loader(self.train, self.train_dir, self.batch_size, (224, 224));

        if model_name != 'resnet' and model_name != 'vgg' and model_name != 'efficient_net':
            raise ValueError("Variable 'model_name' can assume only those values: 'resnet', 'vgg', 'efficient_net'")

        self.model = self.generate_model()

    def check_gpu():
        print(tensorflow.config.list_physical_devices('GPU'))

    def extract_labels(batch_y):
        labels = []
        for i, x in enumerate(batch_y):
            labels.append(int(x))
        return labels

    def generate_model(self, summary=False):
        model = None

        if self.model_name == 'efficient_net':
            cut_layer_name = 'block7a_se_reduce'
            base_model = keras.applications.EfficientNetB0(include_top=True, weights='imagenet')
            self.preprocess_fun = lambda x: keras.applications.efficientnet.preprocess_input(x)
            model = keras.Model(inputs=base_model.input, outputs=base_model.get_layer(cut_layer_name).output)

        ## resnet definition
        elif self.model_name == 'resnet':
            cut_layer_name = 'conv4_block36_2_relu'
            base_model = keras.applications.ResNet152V2(
                include_top=True,
                weights="imagenet",
                pooling=None,
            )

            model = keras.Model(inputs=base_model.input, outputs=base_model.get_layer(cut_layer_name).output)

            self.preprocess_fun = lambda x: keras.applications.resnet_v2.preprocess_input(x)
            ## vgg definition
        elif self.model_name == 'vgg':

            cut_layer_name = 'fc1'
            base_model = keras.applications.VGG16(
                include_top=True,
                weights="imagenet",
                pooling=None,
            )

            model = keras.Model(inputs=base_model.input, outputs=base_model.get_layer(cut_layer_name).output)
            self.preprocess_fun = lambda x: keras.applications.vgg16.preprocess_input(x)

        if summary:
            model.summary()

        return model

    def get_out_shape(self):
        out_shape = self.model.output.shape
        out_shape = [i for i in out_shape if i is not None]
        out_shape = np.array(out_shape).prod()

        return out_shape

    def fit_train(self):
        out_shape = self.get_out_shape()

        features = np.zeros((len(self.train), out_shape))
        index = 0
        labels = self.train['label'].values
        images_name = self.train['filename'].values

        for batch_num in tqdm(range(self.dl.number_of_batch())):
            (images, _, _) = self.dl.get_batch(batch_num)
            images = self.preprocess_fun(images)
            fe = self.model.predict(images, verbose=0)
            fe = fe.reshape(fe.shape[0], -1)
            for element in fe:
                features[index] = element
                index = index + 1

        return features, labels, images_name

    def fit_pca(self, features, components=411):
        self.pca = PCA(n_components=components)
        self.pca = self.pca.fit(features)
        return self.pca

    def pca_features(self, pca, features):
        return pca.transform(features)

    def analyze_pca_components(self, pca):
        ratio = pca.explained_variance_ratio_
        cum_prob = ratio.cumsum()
        cum_prob = [(x, i) for x, i in enumerate(cum_prob)]
        return cum_prob

    def fit_knn(self, features, labels, use_pca=False):
        knn = KNeighborsClassifier(n_neighbors=15, algorithm="auto")
        if use_pca and self.pca is not None:
            features = self.pca.transform(features)
        knn.fit(features, labels)
        return knn

    def get_cumulative_prob_position(self, prob, threshold=0.6, pos_proportion=0.2):
        indexed = [(x, i) for x, i in enumerate(prob)]
        sor = sorted(indexed, key=lambda x: x[1], reverse=True)
        sum = 0
        pos = []
        pos_limit = len(prob) * pos_proportion
        for position, prob in sor:
            if sum < threshold and pos_limit > 0:
                pos.append(position)
            else:
                break
            sum = sum + prob
            pos_limit = pos_limit - 1
        return pos

    def filter_with_knn(self, knn, use_pca=False, threshold=0.85, pos_proportion=0.1):
        discarded_filenames = []
        for batch_num in tqdm(range(self.dl.number_of_batch())):
            (images, labels, filenames) = self.dl.get_batch(batch_num)
            images = self.preprocess_fun(images)
            fe = self.model.predict(images, verbose=0)
            fe = fe.reshape(fe.shape[0], -1)
            if use_pca and self.pca is not None:
                fe = self.pca.transform(fe)

            probs = knn.predict_proba(fe)
            predicted = []
            for prob in probs:
                predicted.append(self.get_cumulative_prob_position(prob, threshold, pos_proportion))

            result = [True if x in y else False for x, y in zip(labels, predicted)]
            discarded = [filename if x else None for filename, x in zip(filenames, result)]
            discarded_filenames = discarded_filenames + discarded
        
        return discarded_filenames;
