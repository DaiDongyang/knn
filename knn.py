import numpy as np
import os
import time
import KDTree
from os.path import join
from collections import defaultdict
from tqdm import tqdm

# input file suffix
inf_suffix = '.txt'

trains_cov_pinv = None
kd_tree = None


def get_trains_conv_pinv(train_samples):
    global trains_cov_pinv
    if trains_cov_pinv is None:
        train_T = train_samples.T
        trains_cov = np.cov(train_T)
        trains_cov_pinv = np.linalg.pinv(trains_cov)
        # train_m = get_ints_m_trans_matrix(train_samples)
        # trains_cov_pinv = train_m.dot(train_m.T)
    return trains_cov_pinv


def get_kd_tree(train_samples):
    global kd_tree
    if kd_tree is None:
        # print(train_samples.dtype)
        m, _ = train_samples.shape
        ls = np.array(list(range(m)))
        kd_tree = KDTree.construct_kd_tree(train_samples, ls)
    return kd_tree


def release_cache():
    global trains_cov_pinv
    global kd_tree
    trains_cov_pinv = None
    kd_tree = None


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


def pca_trans_with_new_d(train_samples, test_samples, new_d):
    miu = np.average(train_samples, axis=0)
    avg_trains = train_samples - miu
    cov = np.cov(avg_trains.T)
    c_roots, Q = np.linalg.eig(cov)
    c_roots = c_roots.real
    Q = Q.real
    # print(c_roots.dtype)
    indexed_c_roots = zip(c_roots, range(len(c_roots)))
    sorted_indexed_c_roots = sorted(indexed_c_roots, reverse=True, key=lambda x: x[0])
    indexs = [item[1] for item in sorted_indexed_c_roots[0:new_d]]
    trains_m = Q[:, indexs]
    # print(trains_m.dtype)
    new_trains = avg_trains.dot(trains_m)
    new_test = (test_samples - miu).dot(trains_m)
    return new_trains, new_test, new_d


def pca_trans_with_threshold(train_samples, test_samples, threshold):
    miu = np.average(train_samples, axis=0)
    avg_trains = train_samples - miu
    cov = np.cov(avg_trains.T)
    c_roots, Q = np.linalg.eig(cov)
    c_roots = c_roots.real
    Q = Q.real
    indexed_c_roots = zip(c_roots, range(len(c_roots)))
    sorted_indexed_c_roots = sorted(indexed_c_roots, reverse=True, key=lambda x: x[0])
    t = (np.sum(c_roots)) * threshold
    indexs = []
    sum = 0
    for item in sorted_indexed_c_roots:
        sum += item[0]
        indexs.append(item[1])
        if sum > t:
            break
    trains_m = Q[:, indexs]
    new_trains = avg_trains.dot(trains_m)
    new_test = (test_samples - miu).dot(trains_m)
    new_d = len(indexs)
    return new_trains, new_test, new_d


# get k nearest euclidean distance index to a new instance with training sample
def get_knn_e_dist(train_samples, new_inst, k):
    diff_sq_matrix = (train_samples - new_inst) ** 2
    dist_sq = np.sum(diff_sq_matrix, axis=1)
    idx = np.argpartition(dist_sq, k)
    k_idx = idx[0:k]
    # return the index and distance
    return k_idx, np.sqrt(dist_sq[k_idx])


def get_knn_e_dist_with_kdtree(train_samples, new_inst, k):
    kdtree = get_kd_tree(train_samples)
    knn_heaps = KDTree.find_knn_from_kd_tree(new_inst, kdtree, k)
    idx = np.array([item[2] for item in knn_heaps.knns[1:]])
    dist_sq = np.array([item[0] for item in knn_heaps.knns[1:]])
    # print(idx)
    # print(dist_sq)
    return idx, np.sqrt(dist_sq)


# method for calculate Mahalay distance before optimization
def get_knn_m_dist(train_samples, inst, k):
    pinv = get_trains_conv_pinv(train_samples)
    diff_matrix = train_samples - inst
    dist_sq = np.diag(np.dot(np.dot(diff_matrix, pinv), diff_matrix.T))
    idx = np.argpartition(dist_sq, k)
    k_idx = idx[0:k]
    return k_idx, np.sqrt(dist_sq[k_idx])


# def get_ints_m_trans_matrix(train_samples):
#     train_T = train_samples.T
#     trains_cov = np.cov(train_T)
#     ksi, Q = np.linalg.eig(trains_cov)
#     pinv_ksi = np.linalg.pinv(np.diag(ksi))
#     pinv_ksi_sqrt = np.sqrt(pinv_ksi)
#     trans_m = np.dot(Q, pinv_ksi_sqrt)
#     # test = (np.dot(trans_m, trans_m.T)**2 - np.linalg.pinv(trains_cov)**2)**2
#     # print(np.sum(test.flat))
#     return trans_m


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
    if max_dist == min_dist:
        return get_label_by_knn(train_ls, k_idx, k_dist)
    for idx, dist in zip(k_idx, k_dist):
        counter[train_ls[idx]] += (max_dist - dist) / (max_dist - min_dist)
    l, _ = max(counter.items(), key=lambda x: x[1])
    return l


def get_test_samples_labels(k, train_samples, train_ls, test_samples, get_knn_func, get_label_func, pca_parameter):
    _, dims = train_samples.shape
    if not (pca_parameter is None) and pca_parameter > 0:
        if pca_parameter < 1:
            train_samples, test_samples, dims = pca_trans_with_threshold(train_samples, test_samples, pca_parameter)
        else:
            train_samples, test_samples, dims = pca_trans_with_new_d(train_samples, test_samples, pca_parameter)
    result = []
    for inst in tqdm(test_samples):
        k_idx, k_dist = get_knn_func(train_samples, inst, k)
        l = get_label_func(train_ls, k_idx, k_dist)
        result.append(l)
    release_cache()
    return np.array(result).transpose(), dims


def result_evaluate(g_ls, r_ls):
    g_ls = g_ls.reshape(-1, 1)
    r_ls = r_ls.reshape(-1, 1)
    check = np.array(g_ls == r_ls)
    total_num = g_ls.size
    acc = np.sum(check) / total_num
    labels = np.arange(10)
    positives = np.zeros(labels.size)
    trues = np.zeros(labels.size)
    tps = np.zeros(labels.size)
    for i in range(total_num):
        positives[r_ls[i]] += 1
        trues[g_ls[i]] += 1
        # print(r_ls.shape)
        # print(g_ls.shape)
        # print(i)
        # print(r_ls[i])
        # print(g_ls[i])
        if r_ls[i] == g_ls[i]:
            tps[r_ls[i]] += 1
    pre = tps/positives
    rec = tps/trues
    F1 = (2 * pre * rec)/(pre + rec)
    macro_pre = np.average(pre)
    macro_rec = np.average(rec)
    macro_F1 = (2 * macro_pre * macro_rec)/(macro_pre + macro_rec)
    return acc, pre, rec, F1, macro_pre, macro_rec, macro_F1


if __name__ == '__main__':
    train_dir = './digits/trainingDigits'
    test_dir = './digits/testDigits'
    k_value = 3
    get_knn_function = get_knn_e_dist_with_kdtree  # or get_k_min_e_dist_with_kd_tree, or get_k_min_m_dist
    get_label_function = get_label_by_knn  # or get_label_by_wknn
    pca_param = 0

    train_set, train_labels = load_sample_set(train_dir)
    test_set, ground_labels = load_sample_set(test_dir)

    print('processing...')
    start_time = time.time()
    result_labels, d = get_test_samples_labels(k_value, train_set, train_labels, test_set, get_knn_function, get_label_function,
                                               pca_param)
    end_time = time.time()
    elapsed = end_time - start_time

    accuracy, precision, recall, F_1, macro_precision, macro_recall, macro_F_1 = result_evaluate(ground_labels, result_labels)
    print('accuracy =', accuracy)
    print('precision =', precision)
    print('recall =', recall)
    print('F1 = ', F_1)
    print('macro_precision =', macro_precision)
    print('macro_recall =', macro_recall)
    print('macro_F1 =', macro_F_1)
    print('execution time =', elapsed)
    print('dimension =', d)
