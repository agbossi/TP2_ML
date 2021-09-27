from Tree import Tree
import random
import pandas as pd


def random_forest(data_set, sample_size, num_trees):
    forest = []
    for i in range(num_trees):
        sample = get_sample(data_set, sample_size)
        decision_tree = Tree()
        forest.append(decision_tree.train(sample))
    return forest


def get_sample(data_set, sample_size):
    sampler = {}
    for key in data_set.keys():
        sampler[key] = []
    for i in range(sample_size):
        random_element = get_element(data_set, sample_size)
        for key in random_element.keys():
            sampler[key].append(random_element[key])
    data_items = sampler.items()
    data_list = list(data_items)
    sample = pd.DataFrame(data_list)
    return sample


def get_element(data_set, sample_size):
    element = {}
    index = random.choice(range(sample_size))
    for key in data_set.keys():
        element[key] = data_set[key][index]
    return element
