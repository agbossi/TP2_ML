import math
import os
import numpy as np
import pandas as pd

relevant = ['wordcount', 'titleSentiment', 'sentimentValue', 'Star Rating']
objective = ['Star Rating']
objective_classes = [1, 2, 3, 4, 5]
data = pd.read_csv('Data/reviews_sentiment.csv', delimiter=";")
for i, row in data.iterrows():
    wordcount = len(row['Review Text'].split(' '))
    if wordcount != row['wordcount']:
        data.at[i, 'wordcount'] = wordcount
data = data.dropna()
data = data[relevant]

ej1data = data[data['Star Rating'] == 1]

print(ej1data)
mean = 0
for i, row in ej1data.iterrows():
    mean = mean + ej1data.at[i, 'wordcount']
# mean = mean / ej1data.count()
mean = mean/len(ej1data.index)
print(mean)


# print(sentiments.star_rating.unique())

labels = np.array(data.iloc[:, 3])

print(objective_classes)
print(labels)



