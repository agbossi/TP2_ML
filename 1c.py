import os
import pandas as pd
import RandomForest as rf
from Data_resamplers import train_test_split
from Metrics import ConfusionMatrix
import Tree


def print_confusion_matrix(confusion_matrix):
    confusion_matrix.summarize()
    confusion_matrix.print_confusion_matrix()
    print('s')
    print(confusion_matrix.get_all_s())
    print('recalls')
    print(confusion_matrix.get_recalls())
    print('precisions')
    print(confusion_matrix.get_precisions())
    print('accuracies')
    print(confusion_matrix.get_accuracies())
    print('f1')
    print(confusion_matrix.get_f1_scores())


def get_forest_confusion_matrix(forest, test_set):
    votes_p, votes_n = 0, 0
    confusion_matrix = ConfusionMatrix([0, 1])
    class_col = forest[0].get_class_column()
    for test_element in test_set:
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


path = os.path.abspath('Data/german_credit_adjusted.csv')
df = pd.read_csv(path)

training_percent = 0.7
sets = train_test_split(df, training_percent)[0]
num_trees = 3
sample_size = 300
forest = rf.random_forest(sets[0], sample_size, num_trees)
conf_matrix = get_forest_confusion_matrix(forest, sets[1])
print_confusion_matrix(conf_matrix)