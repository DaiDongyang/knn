class Node:
    def __init__(self):
        self.min_array = []
        self.max_array = []
        self.cur_d = -1
        self.instance = []
        self.father = None
        self.left_child = None
        self.right_child = None


def construct_kdtree(samples, d_count):
    if samples.size == 0:
        return None

    return None

