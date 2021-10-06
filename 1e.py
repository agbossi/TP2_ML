import os
import matplotlib.pyplot as plt
import pandas as pd
from Tree import Tree
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


def get_tree_confusion_matrix(tree, test_set):
    classifications = tree.test(test_set)
    confusion_matrix = ConfusionMatrix([0, 1])
    for elem_index, classification in classifications.items():
        real_classification = test_set.iloc[1, :][tree.get_class_column()]
        confusion_matrix.add_entry(real_classification, classification)
    return confusion_matrix


def get_precision_points(tree, train, test, heights):
    x_points = []
    y_points_test = []
    y_points_train = []
    for height in range(1, heights+1):
        tree.train(data_set=train, max_depth=height, min_elements_for_fork=3)
        x_points.append(tree.get_node_amount())

        conf_matrix = get_tree_confusion_matrix(tree, train)
        y_points_train.append(conf_matrix.get_s())

        conf_matrix = get_tree_confusion_matrix(tree, test)
        y_points_test.append(conf_matrix.get_s())
    return x_points, y_points_train, y_points_test


def get_forest_precision_points(training_set, test_set, heights, num_trees):
    x_points = []
    y_train_f, y_test_f = [], []
    for height in range(1, heights+1):
        forest = rf.random_forest(training_set, sample_size, num_trees, max_depth=height, min_elements_for_fork=3)
        x_points.append(int(rf.avg_height(forest)))

        matrix = get_forest_confusion_matrix(forest, test_set)
        y_test_f.append(matrix.get_s())

        matrix = get_forest_confusion_matrix(forest, training_set)
        y_train_f.append(matrix.get_s())

    return x_points, y_train_f, y_test_f


def plot_precision(tree, training_set, test_set, num_trees, heights):
    x_points_f, y_train_f, y_test_f = get_forest_precision_points(training_set, test_set, heights, num_trees)
    x_points, y_train, y_test = get_precision_points(tree, training_set, test_set, heights)

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

heights = 3
num_trees = 3
sample_size = 10
training_percent = 0.7
tree = Tree()
plot_precision(tree, sets[0], sets[1], num_trees, heights)
