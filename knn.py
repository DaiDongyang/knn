import numpy as np
import os
from os.path import join
from collections import defaultdict
from tqdm import tqdm


# input file suffix
inf_suffix = '.txt'

trains_cov_pinv = None


def get_trains_conv_pinv(train_samples):
    global trains_cov_pinv
    if trains_cov_pinv is None:
        train_T = train_samples.T
        trains_cov = np.cov(train_T)
        trains_cov_pinv = np.linalg.pinv(trains_cov)
    return trains_cov_pinv


# load a instance from a file
def load_instance(fold_path, file_name):
    (l, _) = file_name.split('_')
    instance = []
    with open(join(fold_path, file_name), 'r') as inf:
        for line in inf:
            instance += [int(i) for i in line.strip()]
    return instance, int(l)


# load sample set according to a fold path
def load_sample_set(fold_path):
    instances = []
    ls = []
    for file_name in os.listdir(fold_path):
        if file_name.endswith(inf_suffix):
            instance, l = load_instance(fold_path, file_name)
            instances.append(instance)
            ls.append(l)
    return np.array(instances), np.array(ls).transpose()


# get k nearest euclidean distance index to a new instance with training sample
def get_k_min_e_dist(train_samples, new_inst, k):
    # diff = train_samples - new_inst
    # dist_sq = np.diag(np.dot(diff, diff.T))
    diff_sq_matrix = (train_samples - new_inst) ** 2
    dist_sq = np.sum(diff_sq_matrix, axis=1)
    idx = np.argpartition(dist_sq, k)
    k_idx = idx[0:k]
    # return the index and distance
    return k_idx, np.sqrt(dist_sq[k_idx])


# method for calculate Mahalay distance before optimization
def get_k_min_m_dist(train_samples, inst, k):
    pinv = get_trains_conv_pinv(train_samples)
    diff_matrix = train_samples - inst
    dist_sq = np.diag(np.dot(np.dot(diff_matrix, pinv), diff_matrix.T))
    idx = np.argpartition(dist_sq, k)
    k_idx = idx[0:k]
    return k_idx, np.sqrt(dist_sq[k_idx])


def get_ints_m_trans_matrix(train_samples):
    train_T = train_samples.T
    trains_cov = np.cov(train_T)
    ksi, Q = np.linalg.eig(trains_cov)
    pinv_ksi = np.linalg.pinv(np.diag(ksi))
    pinv_ksi_sqrt = np.sqrt(pinv_ksi)
    trans_m = np.dot(Q, pinv_ksi_sqrt)
    test = (np.dot(trans_m, trans_m.T) - np.linalg.pinv(trains_cov))**2
    print(np.sum(test.flat))
    return trans_m


def trans_featrues(train_samples, test_samples, trans_m):
    train_samples = np.dot(train_samples, trans_m)
    test_samples = np.dot(test_samples, trans_m)
    return train_samples, test_samples


# get a label simple by knn method
def get_label_by_knn(train_ls, k_idx, k_dist):
    counter = defaultdict(lambda: 0)
    for idx in k_idx:
        counter[train_ls[idx]] += 1
    l, _ = max(counter.items(), key=lambda x: x[1])
    return l


def get_label_by_wknn(train_ls, k_idx, k_dist):
    counter = defaultdict(lambda: 0)
    min_dist = np.min(k_dist)
    max_dist = np.max(k_dist)
    for idx, dist in zip(k_idx, k_dist):
        counter[train_ls[idx]] += (max_dist - dist) / (max_dist - min_dist)
    l, _ = max(counter.items(), key=lambda x: x[1])
    return l


def get_test_samples_labels(k, train_samples, train_ls, test_samples, get_k_min_func, get_label_func):
    result = []
    # m transform
    trans_m = get_ints_m_trans_matrix(train_samples)
    tran_samples, test_samples = trans_featrues(train_samples, test_samples, trans_m)
    # todo: remove tqdm
    for inst in tqdm(test_samples):
        k_idx, k_dist = get_k_min_func(train_samples, inst, k)
        l = get_label_func(train_ls, k_idx, k_dist)
        result.append(l)
    return np.array(result).transpose()
