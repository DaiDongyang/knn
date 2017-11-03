import KDTree
import numpy as np


def print_kd_tree(node):
    if node is None:
        print("None")
        return
    print(node.instance, node.label)
    print_kd_tree(node.left_child)
    print_kd_tree(node.right_child)


def test_construct_kd_tree():
    samples = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    ls = np.array(list(range(6)))
    kd_tree = KDTree.construct_kd_tree(samples, ls)
    print_kd_tree(kd_tree)


def test_knn_heap():
    capacity = 9

    knn_heap = KDTree.KnnHeap(capacity)
    print(knn_heap.get_max_dist_sq())
    knn_heap.update_value(1, None, None)
    knn_heap.update_value(2, None, None)
    knn_heap.update_value(9, None, None)
    knn_heap.update_value(8, None, None)
    print(knn_heap.get_max_dist_sq())
    knn_heap.update_value(1, None, None)
    knn_heap.update_value(2, None, None)
    knn_heap.update_value(11, None, None)
    knn_heap.update_value(20, None, None)
    knn_heap.update_value(111, None, None)
    knn_heap.update_value(22, None, None)
    print(knn_heap.knns)
    knn_heap.update_value(101, None, None)
    print(knn_heap.knns)
    knn_heap.update_value(181, None, None)
    print(knn_heap.knns)
    knn_heap.update_value(1, None, None)
    print(knn_heap.knns)
    print(knn_heap.get_max_dist_sq())


def test_search_sub_tree():
    capacity = 2
    samples = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    ls = np.array(list(range(6)))
    kd_tree = KDTree.construct_kd_tree(samples, ls)
    knn_heap = KDTree.KnnHeap(capacity)
    KDTree.search_sub_tree((0, 0), kd_tree, knn_heap)
    print(knn_heap.knns)

if __name__ == '__main__':
    # test_construct_kd_tree()
    # test_knn_heap()
    test_search_sub_tree()

