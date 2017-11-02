import numpy as np


class Node:
    def __init__(self, inst, label, min_a, max_a, cur_f, f_node):
        self.instance = inst
        self.label = label
        self.min_array = min_a
        self.max_array = max_a
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
    select_idx = order[int(m/2)]
    instance = samples[select_idx, :]
    left_idxs = order[0: int(m/2)]
    right_idxs = order[int(m/2) + 1:m]
    node = Node(instance, labels[select_idx], min_array, max_array, feature, father_node)
    node.left_child = construct_kd_node(samples[left_idxs, :], labels[left_idxs], feature+1, node)
    node.right_child = construct_kd_node(samples[right_idxs, :], labels[right_idxs], feature+1, node)
    return node


def construct_kd_tree(samples, ls):
    return construct_kd_node(samples, ls, 0, None)


# limit size max heap
class KnnHeap:

    def __init__(self, capacity):
        self.capacity = capacity
        self.next_idx = 0
        # self.knns [(dist_sq, instance, label)]
        self.knns = []

    def get_max_dist_sq(self):
        if self.next_idx == 0:
            return float('inf')
        else:
            return self.knns[0][0]

