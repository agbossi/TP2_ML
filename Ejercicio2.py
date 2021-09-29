import numpy as np
import pandas as pd
<<<<<<< HEAD
import math
=======
from Metrics import ConfusionMatrix
>>>>>>> cb2740f8f76fb585c6027370e3c4ef7ba3c944d1

from sklearn.model_selection import train_test_split

relevant = ['wordcount', 'titleSentiment', 'sentimentValue', 'Star Rating']
objective = ['Star Rating']
features = ['wordcount', 'sentimentValue', 'titleSentiment']
objective_classes = [1, 2, 3, 4, 5]
data = pd.read_csv('Data/reviews_sentiment.csv', delimiter=";")
for i, row in data.iterrows():
    wordcount = len(row['Review Text'].split(' '))
    if wordcount != row['wordcount']:
        data.at[i, 'wordcount'] = wordcount
data = data.dropna()
data = data[relevant]
# print(data)
ej1data = data[data['Star Rating'] == 1]

# print(ej1data)
mean = 0
for i, row in ej1data.iterrows():
    mean = mean + ej1data.at[i, 'wordcount']
mean = mean / len(ej1data.index)
# print(mean)

labels = np.array(data.iloc[:, 3])
data['titleSentiment'].replace('positive', 1, inplace=True)
data['titleSentiment'].replace('negative', 0, inplace=True)
data = data[features]
# print(data)
# print(objective_classes)
# print(labels)
X_train, X_test, y_train, y_test = train_test_split(data.to_numpy(), labels, test_size=0.3)
list_train = list(zip(X_train, y_train))
list_test = list(zip(X_test, y_test))


def weight_func(distance, is_weighted):
    if is_weighted:
        return 1/(distance ** 2) if distance != 0 else math.inf
    else:
        return 1


def classify(training, test, kay, is_weighted):
    output = []
    for o in range(len(test)):
        winner = classify_element(list_train, test[o][0], kay, is_weighted)
        s = 1
        while len(winner) > 1:
            winner = classify_element(list_train, test[o][0], kay + s, is_weighted)
            s = s + 1
        output.append(winner[0])
    return output


def classify_element(training, test_element, kay, is_weighted):
    results = {}
    for k in range(len(training)):
        dist = np.linalg.norm(training[k][0] - test_element)
        results[dist] = training[k][1]
    results = sorted(results.items())
    results = results[:kay]
    values = [item[1] for item in results]
    distances = [item[0] for item in results]
    ret = np.zeros(6)
    for s in range(len(values)):
        ret[values[s]] += weight_func(distances[s], is_weighted)
    max_value = np.max(ret)
    winner = np.where(ret == max_value)[0]
    return winner



classify(list_train, list_test, 5, 1)
print(y_test)
print(" ")
print(" ")
print(" ")
classify(list_train, list_test, 5, 0)
print(y_test)

classifications = classify(list_train, list_test, 5, 1)
confusion_matrix = ConfusionMatrix(['1', '2', '3', '4', '5'])
for i in range(len(classifications)):
    confusion_matrix.add_entry(y_test[i], classifications[i])
confusion_matrix.summarize()
print('recalls')
confusion_matrix.get_recalls()
print('precisions')
confusion_matrix.get_precisions()
print('accuracies')
confusion_matrix.get_accuracies()
print('f1')
confusion_matrix.get_f1_scores()

