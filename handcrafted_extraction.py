import sys
sys.path.append("..")
sys.path.append(".")
import os

from multiprocessing.pool import ThreadPool
from handcrafted import extract_features, color_histograms, sift, gabor_response, lbp
import numpy as np;
from misc import unroll_arrays
import bag_of_words as bow
import pickle
from tqdm import tqdm

def store_features(images_dir, images_names, store_path, kmeans=None, sift_info=True, gabor_obj=True, color=True, lbp_info=True):

    (gabor, color_features, lbp_features, sift_features, kmeans, bow_features, all_features) = compute_features(
        images_dir, images_names, kmeans=kmeans, sift_info=sift_info, gabor_obj=gabor_obj, color=color, lbp_info=lbp_info, separated=True
        );

    gabor_dir = store_path + 'gabor/';
    color_dir = store_path + 'color/';
    lbp_dir = store_path + 'lbp/';
    sift_dir = store_path + 'sift_features/';
    bow_dir = store_path + 'bow_features/';

    if gabor_obj is not None:
        store_feature(gabor, gabor_dir, images_names)
    if color is not None:
        store_feature(color_features, color_dir, images_names)
    if lbp_info is not None:
        store_feature(lbp_features, lbp_dir, images_names)
    if sift_info is not None:
        store_feature(sift_features, sift_dir, images_names)
        store_feature(bow_features, bow_dir, images_names)

        with open(store_path + 'kmeans.pkl', "wb") as f:
            pickle.dump(kmeans, f);

    return (all_features, kmeans)

def load_features(dir, feat_name, filenames):
    sample = None;
    index = 0;
    while sample is None or len(sample.shape) == 0:
        sample = np.load(dir + feat_name + '/' + filenames[index] + ".npy", allow_pickle=True)
    
    all_features = []
    index = 0
    for filename in tqdm(filenames):
        feats = np.load(dir + feat_name + '/' + filename + ".npy", allow_pickle=True)
        all_features.append(feats);

    return all_features;

def load_all_features(dir, filenames, load_sift = False, load_color=True, load_gabor = True):
    gabor = None;
    color = None;
    lbp = None;
    sift = None;
    bow = None;

    all_features = None

    if load_gabor:
        try:
            gabor = load_features(dir, 'gabor', filenames);
            all_features = concat(all_features, gabor, axis=1);
        except:
            print('No gabor features found')

    if load_color:
        try:
            color = load_features(dir, 'color', filenames);
            all_features = concat(all_features, color, axis=1);
        except:
            print('No color features found')

    try:
        lbp = load_features(dir, 'lbp', filenames);
        all_features = concat(all_features, lbp, axis=1);
    except:
        print('No lbp features found')

    if load_sift:
        try:
            sift = load_features(dir, 'sift_features', filenames)
            sift = unroll_arrays(sift)
            all_features = concat(all_features, sift, axis=1);
        except:
            print('No sift features found')

    try:
        bow = load_features(dir, 'bow_features', filenames)
        all_features = concat(all_features, bow, axis=1);
    except:
        print('No bow features found')
    
    with open(dir + 'kmeans.pkl', "rb") as f:
        kmeans = pickle.load(f);

    return (all_features, kmeans);


def store_feature(features, folder, images_names):
    if not os.path.exists(folder):
        os.makedirs(folder);

    for feat, filename in zip(features, images_names):
        np.save(folder + filename, feat)

    

def compute_features(images_dir, images_names, kmeans=None, sift_info=True, gabor_obj=True, color=True, lbp_info=True, separated=False):
    """ 
    Extract various types of features

    :kmeans None or kmeans pre-trained
    :sift_info Boolean or dict with 'max_number' and 'num_of_sample_kmeans', 'voc_size', 'max_iter'
    :gabor_obj Boolean or dictionary with 'angles', 'lambdas', 'gammas'
    :color_histogram Boolean
    :lbp_info Boolean or array of distances
    """
    import time

    features_names = []
    if sift_info is not None:
        features_names.append("sift")
    if gabor_obj is not None:
        features_names.append("gabor")
    if color is not None:
        features_names.append("color")
    if lbp_info is not None:
        features_names.append("lbp")

    print("Preparing to compute: ", ' ,'.join(features_names))
    time.sleep(3)


    pool = ThreadPool(processes=14)

    gabor = None;
    color_features = None;
    lbp_features = None;
    sift_features = None;
    bow_features = None;

    if sift_info == True:
        sift_info = {
            'max_number': 50,
            'num_of_sample_kmeans': 1,
            'voc_size': 300,
            'max_iter': 500
        };
    elif sift_info == False:
        sift_info = None
    
    if gabor_obj == True:
        gabor_obj = {
            'angles': np.arange(0, np.pi, np.pi/4),
            'lambdas': np.arange(0, 1, 0.2),
            'gammas': [0.5]
        }
    elif gabor_obj == False:
        gabor_obj = None;
    
    if lbp_info == True:
        lbp_info = [2];
    elif lbp_info == False:
        lbp_info = None;

    if sift_info is not None:
        sift_features = pool.apply_async(extract_features, (images_dir, images_names,
                                                        [(lambda img: sift(img, max_features=sift_info['max_number']))]))

    if gabor_obj is not None:
        gabor = pool.apply_async(extract_features, (images_dir, images_names,
                                                        [(lambda img: gabor_response(img, (10, 10), gabor_obj['angles'], [5], gabor_obj['lambdas'], gabor_obj['gammas']))]))

    if color:
        color_features = pool.apply_async(extract_features, (images_dir, images_names,
                                                        [(lambda img: color_histograms(img))]))

    if lbp_info is not None:
        lbp_features = pool.apply_async(extract_features, (images_dir, images_names,
                                                        [(lambda img: lbp(img, lbp_info))]))

    all_features = None
    
    if gabor_obj is not None:
        gabor = gabor.get()
        gabor = np.array(gabor)
        all_features = concat(all_features, gabor);
    if color:
        color_features = color_features.get();
        color_features = np.array(color_features);
        color_features = color_features.reshape(color_features.shape[0], -1);
        all_features = concat(all_features, color_features);
    if lbp_info is not None:
        lbp_features = lbp_features.get();
        lbp_features = np.array(lbp_features);
        all_features = concat(all_features, lbp_features);
    if sift_info is not None:
        sift_features = sift_features.get()
        unrolled = unroll_arrays(sift_features, sift_info['num_of_sample_kmeans']);
        if kmeans is None:
            kmeans = bow.fit(unrolled, vocabulary_size=sift_info['voc_size'], verbose=True, n_init=1, max_iter=sift_info['max_iter']);
        bow_features = bow.predict(kmeans, sift_features);
        all_features = concat(all_features, bow_features)

    if separated:
        return (gabor, color_features, lbp_features, sift_features, kmeans, bow_features, all_features)
    return (all_features, kmeans);


def concat(arr1, arr2, axis=1):
    if arr1 is None:
        return np.array(arr2)

    return np.concatenate((arr1, np.array(arr2)), axis=axis)