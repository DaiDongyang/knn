import numpy as np


class Node:
    def __init__(self, inst, label, min_a, max_a, cur_f, f_node):
        self.instance = inst
        self.label = label
        self.arrays = np.empty((2, len(inst)))
        self.arrays[0, :] = min_a
        self.arrays[1, :] = max_a
        self.cur_feature = cur_f
        self.father = f_node
        self.left_child = None
        self.right_child = None


def construct_kd_node(samples, labels, feature, father_node):
    if samples.size == 0:
        return None
    m, d = samples.shape
    feature = feature % d
    min_array = np.min(samples, axis=0)
    max_array = np.max(samples, axis=0)
    order = np.argsort(samples[:, feature])
    select_idx = order[int(m / 2)]
    instance = samples[select_idx, :]
    left_idxs = order[0: int(m / 2)]
    right_idxs = order[int(m / 2) + 1:m]
    node = Node(instance, labels[select_idx], min_array, max_array, feature, father_node)
    node.left_child = construct_kd_node(samples[left_idxs, :], labels[left_idxs], feature + 1, node)
    node.right_child = construct_kd_node(samples[right_idxs, :], labels[right_idxs], feature + 1, node)
    return node


def construct_kd_tree(samples, ls):
    return construct_kd_node(samples, ls, 0, None)


# limit size max heap
class KnnHeap:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cur_max_idx = 0
        # self.knns [(dist_sq, instance, label)]
        self.knns = [(float('inf'), None, None)]

    def get_max_dist_sq(self):
        if self.cur_max_idx < self.capacity:
            return float('inf')
        else:
            return self.knns[1][0]

    def update_value(self, dist_sq, instance, label):
        if self.cur_max_idx < self.capacity:
            self.cur_max_idx += 1
            self.knns.append((dist_sq, instance, label))
            i = self.cur_max_idx
            while i > 0 and dist_sq > self.knns[int(i / 2)][0]:
                self.knns[i] = self.knns[int(i / 2)]
                i = int(i / 2)
            self.knns[i] = (dist_sq, instance, label)
        else:
            self.knns[1] = (dist_sq, instance, label)
            i = 1
            while 2 * i + 1 <= self.cur_max_idx and (
                    self.knns[2 * i][0] > dist_sq or self.knns[2 * i + 1][0] > dist_sq):
                j = (2 * i) if (self.knns[2 * i][0] > self.knns[2 * i + 1][0]) else (2 * i + 1)
                self.knns[i] = self.knns[j]
                i = j
            if 2 * i <= self.cur_max_idx and self.knns[2 * i][0] > dist_sq:
                self.knns[i] = self.knns[2 * i]
                i = 2 * i
            self.knns[i] = (dist_sq, instance, label)


def get_brother(node):
    if node.father is None:
        return None
    if node is node.father.left_child:
        return node.father.right_child
    else:
        return node.father.left_child


def search_sub_tree(sample, sub_tree, knn_heap):
    if sub_tree is None:
        return
    v_d = np.min(np.abs(sub_tree.arrays - sample), axis=0)
    dist_sq = np.dot(v_d, v_d.T)
    if dist_sq > knn_heap.get_max_dist_sq():
        return
    diff = sub_tree.instance - sample
    dist_sq = np.dot(diff, diff.T)
    print(sub_tree.instance)
    if dist_sq < knn_heap.get_max_dist_sq():
        knn_heap.update_value(dist_sq, sub_tree.instance, sub_tree.label)
    print(knn_heap.knns)
    print()
    search_sub_tree(sample, sub_tree.left_child, knn_heap)
    search_sub_tree(sample, sub_tree.right_child, knn_heap)
    return
