import numpy as np


class Node:
    def __init__(self, inst, min_a, max_a, cur_f, f_node):
        self.instance = inst
        self.min_array = min_a
        self.max_array = max_a
        self.cur_feature = cur_f
        self.father = f_node
        self.left_child = None
        self.right_child = None


def construct_kd_node(samples, feature, father_node):
    if samples.size == 0:
        return None
    m, d = samples.shape
    feature = feature % d
    min_array = np.min(samples, axis=0)
    max_array = np.max(samples, axis=0)
    order = np.argsort(samples[:, feature])
    select_idx = order[int(m/2)]
    instance = samples[select_idx, :]
    left_idxs = order[0:int(m/2)]
    right_idxs = order[int(m/2) + 1:m]
    node = Node(instance, min_array, max_array, feature, father_node)
    node.left_child = construct_kd_node(samples[left_idxs], feature+1, node)
    node.right_child = construct_kd_node(samples[right_idxs], feature+1, node)
    return node


def construct_kd_tree(samples):
    return construct_kd_node(samples, 0, None)



