import os

import pandas as pd

from Data_resamplers import train_test_split
from Metrics import ConfusionMatrix
from Tree import Tree
import RandomForest
# from Stat_analysis import histogram as hist

path = os.path.abspath('Data/german_credit_adjusted.csv')
df = pd.read_csv(path)
# df.drop('Duration of Credit (month)', inplace=True, axis=1)
# df.drop('Credit Amount', inplace=True, axis=1)
# df.drop('Value Savings/Stocks', inplace=True, axis=1)
# df.drop('Length of current employment', inplace=True, axis=1)
# df.drop('Instalment per cent', inplace=True, axis=1)
# df.drop('Sex & Marital Status', inplace=True, axis=1)
# df.drop('Guarantors', inplace=True, axis=1)
# df.drop('Duration in Current address', inplace=True, axis=1)
# df.drop('Age (years)', inplace=True, axis=1)
# df.drop('Concurrent Credits', inplace=True, axis=1)
# df.drop('Type of apartment', inplace=True, axis=1)
# df.drop('Most valuable available asset', inplace=True, axis=1)
# df.drop('No of Credits at this Bank', inplace=True, axis=1)
# df.drop('Occupation', inplace=True, axis=1)
# df.drop('No of dependents', inplace=True, axis=1)
# df.drop('Telephone', inplace=True, axis=1)
# df.drop('Foreign Worker', inplace=True, axis=1)

# col = 'Age (years)'
# xticks = [i for i in range(18, 75, 3)]
# hist(pd.DataFrame(data=df[col], columns=[col]), column=col, xticks=xticks)

# class_column = df.pop('Creditability')
# df.insert(len(df.columns), 'Creditability', class_column)


# de [0 a 11]
# df.loc[(df['Duration of Credit (month)'] >= 0) & (df['Duration of Credit (month)'] <= 11), 'Duration of Credit (month)'] = 0
# de [12 a 14]
# df.loc[(df['Duration of Credit (month)'] >= 12) & (df['Duration of Credit (month)'] <= 14), 'Duration of Credit (month)'] = 1
# de [15 a 23]
# df.loc[(df['Duration of Credit (month)'] >= 15) & (df['Duration of Credit (month)'] <= 23), 'Duration of Credit (month)'] = 2
# de (24 a 27]
# df.loc[(df['Duration of Credit (month)'] >= 24) & (df['Duration of Credit (month)'] <= 27), 'Duration of Credit (month)'] = 3
# de (28 a 75]
# df.loc[(df['Duration of Credit (month)'] >= 28) & (df['Duration of Credit (month)'] <= 75), 'Duration of Credit (month)'] = 4

# de [0 a 23]
# df.loc[(df['Age (years)'] >= 0) & (df['Age (years)'] <= 23), 'Age (years)'] = 0
# de [23 a 24]
# df.loc[(df['Age (years)'] >= 23) & (df['Age (years)'] <= 24), 'Age (years)'] = 1
# de [24 a 25]
# df.loc[(df['Age (years)'] >= 24) & (df['Age (years)'] <= 25), 'Age (years)'] = 2
# de [25 a 26]
# df.loc[(df['Age (years)'] >= 25) & (df['Age (years)'] <= 26), 'Age (years)'] = 3
# de [27 a 28]
# df.loc[(df['Age (years)'] >= 27) & (df['Age (years)'] <= 28), 'Age (years)'] = 4
# de [29 a 30]
# df.loc[(df['Age (years)'] >= 29) & (df['Age (years)'] <= 30), 'Age (years)'] = 5
# de [31 a 32]
# df.loc[(df['Age (years)'] >= 31) & (df['Age (years)'] <= 32), 'Age (years)'] = 6
# de [33 a 35]
# df.loc[(df['Age (years)'] >= 33) & (df['Age (years)'] <= 35), 'Age (years)'] = 7
# de [36 a 38]
# df.loc[(df['Age (years)'] >= 36) & (df['Age (years)'] <= 38), 'Age (years)'] = 8
# de [39 a 42]
# df.loc[(df['Age (years)'] >= 39) & (df['Age (years)'] <= 42), 'Age (years)'] = 9
# de [43 a 48]
# df.loc[(df['Age (years)'] >= 43) & (df['Age (years)'] <= 48), 'Age (years)'] = 10
# de [49 a 75]
# df.loc[(df['Age (years)'] >= 49) & (df['Age (years)'] <= 75), 'Age (years)'] = 11

# de [0 a 1000)
# df.loc[(df['Credit Amount'] >= 0) & (df['Credit Amount'] < 1000), 'Credit Amount'] = 0
# de [1000 a 1500)
# df.loc[(df['Credit Amount'] >= 1000) & (df['Credit Amount'] < 1500), 'Credit Amount'] = 1
# de [1500 a 2000)
# df.loc[(df['Credit Amount'] >= 1500) & (df['Credit Amount'] < 2000), 'Credit Amount'] = 2
# de [2000 a 2500)
# df.loc[(df['Credit Amount'] >= 2000) & (df['Credit Amount'] < 2500), 'Credit Amount'] = 3
# de [2500 a 3000)
# df.loc[(df['Credit Amount'] >= 2500) & (df['Credit Amount'] < 3000), 'Credit Amount'] = 4
# de [3000 a 4000)
# df.loc[(df['Credit Amount'] >= 3000) & (df['Credit Amount'] < 4000), 'Credit Amount'] = 5
# de [4000, -)
# df.loc[df['Credit Amount'] >= 4000, 'Credit Amount'] = 6

# df.to_csv('german_credit_adjusted.csv', index=False)

training_percent = 0.7
sets = train_test_split(df, training_percent)[0]
decision_tree = Tree()
decision_tree.train(sets[0])

forest = RandomForest.random_forest(sets[0], 50, 50)
classifications = decision_tree.test(sets[1])
confusion_matrix = ConfusionMatrix([0, 1])

for i in range(len(sets[1])):
    test_element = sets[1].iloc[i, :]
    confusion_matrix.add_entry(test_element[decision_tree.get_class_column()], classifications[i])







