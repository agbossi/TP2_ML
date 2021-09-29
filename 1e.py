import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import Tree
import RandomForest as rf
from Metrics import ConfusionMatrix
from Data_resamplers import train_test_split


def get_forest_confusion_matrix(forest, test_set):
    confusion_matrix = ConfusionMatrix([0, 1])
    class_col = forest[0].get_class_column()
    for i in range(len(test_set)):
        test_element = test_set.iloc[i, :]
        votes_p, votes_n = 0, 0
        for tree in forest:
            classification = tree.traverse_tree(tree.root, test_element, Tree.INITIAL_DEPTH)
            if classification == 1:
                votes_p += 1
            else:
                votes_n += 1

        forest_classification = 0
        if votes_p > votes_n:
            forest_classification = 1
        confusion_matrix.add_entry(test_element.loc[class_col], forest_classification)
    return confusion_matrix


def get_tree_confusion_matrix(tree, test_set, height=sys.maxsize):
    classifications = tree.test(test_set, height)
    confusion_matrix = ConfusionMatrix([0, 1])
    for elem_index, classification in classifications.items():
        real_classification = test_set.loc[elem_index, tree.get_class_column()]
        confusion_matrix.add_entry(real_classification, classification)
    return confusion_matrix


def get_precision_points(tree, train, test):
    x_points = list(range(tree.get_tree_depth() + 1))
    y_points_test = []
    y_points_train = []
    for height in x_points:
        conf_matrix = get_tree_confusion_matrix(tree, train, height)
        y_points_train.append(conf_matrix.get_s())

        conf_matrix = get_tree_confusion_matrix(tree, test, height)
        y_points_test.append(conf_matrix.get_s())
    return x_points, y_points_train, y_points_test


def get_forest_precision_points(forest, training_set, test_set):
    x_points = list(range(1, len(forest) + 1))
    y_train_f, y_test_f = [], []
    for num_trees in x_points:
        matrix = get_forest_confusion_matrix(forest[0:num_trees], test_set)
        y_test_f.append(matrix.get_s())
        matrix = get_forest_confusion_matrix(forest[0:num_trees], training_set)
        y_train_f.append(matrix.get_s())
    return x_points, y_train_f, y_test_f


def plot_precision(forest, training_set, test_set):
    x_points_f, y_train_f, y_test_f = get_forest_precision_points(forest, training_set, test_set)
    x_points, y_train, y_test = get_precision_points(forest[0], training_set, test_set)

    fig1, ax1 = plt.subplots()
    plt.ylim(0.5, 1.0)
    ax1.plot(x_points, y_train, '-x', color='gray')
    ax1.plot(x_points, y_test, '-x', color='green')
    print(y_test[-1])
    ax1.legend(['Entrenamiento', 'Testeo'])

    fig2, ax2 = plt.subplots()
    plt.ylim(0.5, 1.0)
    ax2.plot(x_points, y_train_f, '-x', color='gray')
    ax2.plot(x_points, y_test_f, '-x', color='green')
    print(y_test_f[-1])
    ax2.legend(['Entrenamiento', 'Testeo'])


path = os.path.abspath('Data/german_credit_adjusted.csv')
df = pd.read_csv(path)

training_percent = 0.7
sets = train_test_split(df, training_percent)[0]

num_trees = 40
sample_size = 500
training_percent = 0.7
forest = rf.random_forest(sets[0], sample_size, num_trees)
plot_precision(forest, sets[0], sets[1])
