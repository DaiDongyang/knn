import os
from os.path import isfile, join

# train_path = './digits/trainingDigits'
# test_path = './digits/testDigits'
# for file in os.listdir(test_path):
#     if isfile(join(test_path, file)) and file.endswith('.txt'):
#         print(join(test_path, file))


# todo: 看一下这里哪写错了，感觉可能是在算协方差矩阵那一块
# def get_k_min_m_dist(train_samples, new_inst, k):
#     samples = np.vstack((train_samples, new_inst)).transpose()
#     cov = np.cov(samples)
#     inv_conv = np.linalg.pinv(cov)
#     diff_matrix = train_samples - new_inst
#     matrix_tmp = np.dot(diff_matrix, inv_conv)
#     dist_sq = []
#     for i in range(len(matrix_tmp)):
#         single_dist_sq = np.dot(matrix_tmp[i], diff_matrix[i])
#         dist_sq.append(single_dist_sq)
#     # print(diff_matrix.shape)
#     # print(inv_conv.shape)
#     # dist_sq = np.diag(np.dot(np.dot(diff_matrix, inv_conv), diff_matrix.T))
#     dist_sq = np.array(dist_sq)
#     idx = np.argpartition(dist_sq, k)
#     k_idx = idx[0:k]
#     # return the index and distance
#     return k_idx, np.sqrt(dist_sq[k_idx])
