import os
import pandas as pd
from Data_resamplers import train_test_split
from Metrics import ConfusionMatrix
from Tree import Tree


def print_confusion_matrix(confusion_matrix):
    confusion_matrix.summarize()
    confusion_matrix.print_confusion_matrix()
    print('s')
    print(confusion_matrix.get_s())
    print('recalls')
    print(confusion_matrix.get_recalls())
    print('precisions')
    print(confusion_matrix.get_precisions())
    print('accuracies')
    print(confusion_matrix.get_accuracies())
    print('f1')
    print(confusion_matrix.get_f1_scores())


path = os.path.abspath('Data/german_credit_adjusted.csv')
df = pd.read_csv(path)

training_percent = 0.7
sets = train_test_split(df, training_percent)[0]
decision_tree = Tree()
decision_tree.train(data_set=sets[0], max_depth=7, min_elements_for_fork=25)

classifications = decision_tree.test(sets[1])
confusion_matrix = ConfusionMatrix([0, 1])

for i in range(len(sets[1])):
    test_element = sets[1].iloc[i, :]
    if classifications[i] is not None:
        confusion_matrix.add_entry(test_element[decision_tree.get_class_column()], classifications[i])
print_confusion_matrix(confusion_matrix)
