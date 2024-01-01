
from multiprocessing.pool import ThreadPool
from handcrafted import extract_features, color_histograms, sift, gabor_response, lbp
import numpy as np;
from misc import unroll_arrays
import bag_of_words as bow

def extract_features(images_dir, images_names, kmeans=None, sift_info=True, gabor_obj=True, color=True, lbp_info=True):
    """ 
    Extract various types of features

    :kmeans None or kmeans pre-trained
    :sift_info Boolean or dict with 'max_number' and 'num_of_sample_kmeans', 'voc_size', 'max_iter'
    :gabor_obj Boolean or dictionary with 'angles', 'lambdas', 'gammas'
    :color_histogram Boolean
    :lbp_info Boolean or array of distances
    """
    pool = ThreadPool(processes=14)

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
        all_features = concat(all_features, np.array(gabor));
    if color:
        color_features = color_features.get();
        color_features = np.array(color_features);
        color_features = color_features.reshape(color_features.shape[0], -1);
        all_features = concat(all_features, color_features);
    if lbp_info is not None:
        lbp_features = lbp_features.get();
        all_features = concat(all_features, np.array(lbp_features));
    if sift_features is not None:
        sift_features = sift_features.get()
        unrolled = unroll_arrays(sift_features, sift_info['num_of_sample_kmeans']);
        if kmeans is None:
            kmeans = bow.fit(unrolled, vocabulary_size=sift_info['voc_size'], verbose=True, n_init=1, max_iter=sift_info['max_iter']);
        bow_features = bow.predict(kmeans, sift_features);
        all_features = concat(all_features, bow_features)


    return (all_features, kmeans);


def concat(arr1, arr2):
    if arr1 is None:
        return arr2

    return np.concatenate((arr1, arr2), axis=1)