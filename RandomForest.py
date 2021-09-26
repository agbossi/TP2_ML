from Tree import Tree
import random


def random_forest(data_set, sample_size, num_trees, ):
    forest = []
    for i in range(num_trees):
        sample = get_sample(data_set, sample_size)
        # forest.append(Tree.build_tree(None, sample, 0)) -> revisar, recibe dict ?
    return forest


def get_sample(data_set, sample_size):
    sample = {}
    for key in data_set.keys():
        sample[key] = []
    for i in range(sample_size):
        random_element = get_element(data_set, sample_size)
        for key in random_element.keys():
            sample[key].append(random_element[key])
    return sample


def get_element(data_set, sample_size):
    element = {}
    index = random.choice(range(sample_size))
    for key in data_set.keys():
        element[key] = data_set[key][index]
    return element
