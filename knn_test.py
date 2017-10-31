import knn
from tqdm import tqdm
from collections import Counter
import numpy as np


def test_load_instance():
    train_dir = './digits/trainingDigits'
    fold_name = '0_0.txt'
    instance, l = knn.load_instance(train_dir, fold_name)
    print(instance)
    print(l)


def test_load_sample_set():
    # np.set_printoptions(threshold='nan')
    train_dir = './digits/trainingDigits'
    instances, ls = knn.load_sample_set(train_dir)
    print(len(instances))
    print()
    print(instances.size)
    print()
    print(ls)
    # print(ls.shape)
    # print(np.max(ls))
    # print(ls[0:10])


def test_several_instance_knn():
    k = 7
    test_num = 100
    train_dir = './digits/trainingDigits'
    train_sample, train_ls = knn.load_sample_set(train_dir)
    test_dir = './digits/testDigits'
    test_fs = ['5_' + str(i) + '.txt' for i in range(test_num)]
    result = []
    for test_f in tqdm(test_fs):
        inst, l = knn.load_instance(test_dir, test_f)
        k_idx, k_dist = knn.get_k_min_m_dist(train_sample, inst, k)
        l = knn.get_label_by_knn(train_ls, k_idx, k_dist)
        result.append(l)
    print(result)


def test_get_test_samples_labels():
    k = 7
    train_dir = './digits/trainingDigits'
    train_samples, train_ls = knn.load_sample_set(train_dir)
    test_dir = './digits/testDigits'
    test_samples, test_ls = knn.load_sample_set(test_dir)
    test_results = knn.get_test_samples_labels(k, train_samples, train_ls, test_samples,
                                               get_k_min_func=knn.get_k_min_e_dist, get_label_func=knn.get_label_by_wknn)
    check = np.array(test_ls.reshape(-1, 1) == test_results.reshape(-1, 1))
    trues = np.sum(check)
    all = check.size
    print(trues)
    print(all)


if __name__ == '__main__':
    # test_load_instance()
    # test_load_sample_set()
    # test_several_instance_knn()
    test_get_test_samples_labels()
