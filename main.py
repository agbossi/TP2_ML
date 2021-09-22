import os

import pandas as pd

path = os.path.abspath('Data/german_credit.csv')
df = pd.read_csv(path)
df.drop('Duration of Credit (month)', inplace=True)
df.drop('Credit Amount', inplace=True)
df.drop('Value Savings/Stocks', inplace=True)
df.drop('Length of current employment', inplace=True)
df.drop('Instalment per cent', inplace=True)
df.drop('Sex & Marital Status', inplace=True)
df.drop('Guarantors', inplace=True)
df.drop('Duration in Current address', inplace=True)
df.drop('Age (years)', inplace=True)
df.drop('Concurrent Credits', inplace=True)
df.drop('Type of apartment', inplace=True)
df.drop('Most valuable available asset', inplace=True)
df.drop('No of Credits at this Bank', inplace=True)
df.drop('Occupation', inplace=True)
df.drop('No of dependents', inplace=True)
df.drop('Telephone', inplace=True)
df.drop('No of dependents', inplace=True)
df.drop('Foreign Worker', inplace=True)







