import os

import pandas as pd

from Data_resamplers import train_test_split
from Tree import Tree

path = os.path.abspath('Data/german_credit.csv')
df = pd.read_csv(path)
df.drop('Duration of Credit (month)', inplace=True, axis=1)
df.drop('Credit Amount', inplace=True, axis=1)
df.drop('Value Savings/Stocks', inplace=True, axis=1)
df.drop('Length of current employment', inplace=True, axis=1)
df.drop('Instalment per cent', inplace=True, axis=1)
df.drop('Sex & Marital Status', inplace=True, axis=1)
df.drop('Guarantors', inplace=True, axis=1)
df.drop('Duration in Current address', inplace=True, axis=1)
df.drop('Age (years)', inplace=True, axis=1)
df.drop('Concurrent Credits', inplace=True, axis=1)
df.drop('Type of apartment', inplace=True, axis=1)
df.drop('Most valuable available asset', inplace=True, axis=1)
df.drop('No of Credits at this Bank', inplace=True, axis=1)
df.drop('Occupation', inplace=True, axis=1)
df.drop('No of dependents', inplace=True, axis=1)
df.drop('Telephone', inplace=True, axis=1)
df.drop('Foreign Worker', inplace=True, axis=1)

class_column = df.pop('Creditability')
df.insert(len(df.columns), 'Creditability', class_column)

training_percent = 0.7
sets = train_test_split(df, training_percent)[0]
decision_tree = Tree()
decision_tree.train(sets[0])
decision_tree.test(sets[1])








