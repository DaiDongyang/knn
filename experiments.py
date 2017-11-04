import knn
import time
import gc
import numpy as np

if __name__ == '__main__':
    experiment_start_t = time.time()
    train_dir = './digits/trainingDigits'
    test_dir = './digits/testDigits'
    result_file = open('result.txt', 'w')
    train_samples, train_ls = knn.load_sample_set(train_dir)
    test_samples, ground_ls = knn.load_sample_set(test_dir)
    ks = [3, 5, 7, 9]
    # ks = [3, 5, 7, 9, 11, 13]
    get_knn_funcs = [knn.get_knn_e_dist, knn.get_knn_e_dist_with_kdtree, knn.get_knn_m_dist]
    get_knn_funcs_str = ['get_knn_e_dist', 'get_knn_e_dist_with_kdtree', 'get_knn_m_dist']
    # get_knn_funcs = [knn.get_knn_e_dist]
    # get_knn_funcs_str = ['get_knn_e_dist']
    get_label_funcs = [knn.get_label_by_knn, knn.get_label_by_wknn]
    get_label_funcs_str = ['get_label_by_knn', 'get_label_by_wknn']
    # get_label_funcs = [knn.get_label_by_wknn]
    # get_label_funcs_str = ['get_label_by_wknn']
    pca_parameters = [0, 4, 8, 16, 32, 64, 128, 256]
    # pca_parameters = [16]
    results = [
        ['k', 'get_knn_function', 'get_label_function', 'pca_parameters', 'accuracy', 'macro_precision', 'macro_recall',
         'macro_F1', 'execution_time', 'd']]
    for k in ks:
        for get_knn_func, i1 in zip(get_knn_funcs, range(3)):
            for get_label_func, i2 in zip(get_label_funcs, range(2)):
                for pca_parameter in pca_parameters:
                    single_result = []
                    print(file=result_file)
                    print('******** Conditions ******', file=result_file)
                    print('k =', k, file=result_file)
                    single_result.append(k)
                    print('get_knn_function =', get_knn_funcs_str[i1], file=result_file)
                    single_result.append(get_knn_funcs_str[i1])
                    print('get_label_function =', get_label_funcs_str[i2], file=result_file)
                    single_result.append(get_label_funcs_str[i2])
                    print('pca_parameters =', pca_parameter, file=result_file)
                    single_result.append(pca_parameter)
                    print('**************************', file=result_file)
                    train_set = np.copy(train_samples)
                    test_set = np.copy(test_samples)
                    train_labels = np.copy(train_ls)
                    ground_labels = np.copy(ground_ls)
                    start_time = time.time()
                    result_labels, d = knn.get_test_samples_labels(k, train_set, train_labels, test_set, get_knn_func,
                                                                   get_label_func, pca_parameter)
                    end_time = time.time()
                    elapsed = end_time - start_time
                    accuracy, precision, recall, F_1, macro_precision, macro_recall, macro_F_1 = knn.result_evaluate(
                        ground_labels, result_labels)
                    print('accuracy =', accuracy, file=result_file)
                    single_result.append(accuracy)
                    print('precision =', precision, file=result_file)
                    # single_result.append(precision)
                    print('recall =', recall, file=result_file)
                    # single_result.append(recall)
                    print('F1 = ', F_1, file=result_file)
                    print('macro_precision =', macro_precision, file=result_file)
                    single_result.append(macro_precision)
                    print('macro_recall =', macro_recall, file=result_file)
                    single_result.append(macro_recall)
                    print('macro_F1 =', macro_F_1, file=result_file)
                    single_result.append(macro_F_1)
                    print('execution time =', elapsed, file=result_file)
                    single_result.append(elapsed)
                    print('dimension =', d, file=result_file)
                    single_result.append(d)
                    print(file=result_file)
                    results.append(single_result)
                    gc.collect()
    experiment_end_t = time.time()
    print('***********************', file=result_file)
    print('Total Time =', experiment_end_t - experiment_start_t, file=result_file)
    print('***********************', file=result_file)
    result_file.close()
    with open('result_table.csv', 'w') as outf:
        for ele in results:
            str_arrays = [str(i) for i in ele]
            ele_str = ', '.join(str_arrays)
            print(ele_str, file=outf)
