import KDTree
import numpy as np


def print_kd_tree(node):
    if node is None:
        print("None")
        return
    print(node.instance)
    print_kd_tree(node.left_child)
    print_kd_tree(node.right_child)


def test_construct_kd_tree():
    samples = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    kd_tree = KDTree.construct_kd_tree(samples)
    print_kd_tree(kd_tree)


if __name__ == '__main__':
    test_construct_kd_tree()


