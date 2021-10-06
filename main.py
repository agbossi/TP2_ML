import os
import pandas as pd

path = os.path.abspath('Data/german_credit.csv')
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

class_column = df.pop('Creditability')
df.insert(len(df.columns), 'Creditability', class_column)


# de [0 a 11]
df.loc[(df['Duration of Credit (month)'] >= 0) & (df['Duration of Credit (month)'] <= 11), 'Duration of Credit (month)'] = 0
# de [12 a 14]
df.loc[(df['Duration of Credit (month)'] >= 12) & (df['Duration of Credit (month)'] <= 14), 'Duration of Credit (month)'] = 1
# de [15 a 23]
df.loc[(df['Duration of Credit (month)'] >= 15) & (df['Duration of Credit (month)'] <= 23), 'Duration of Credit (month)'] = 2
# de (24 a 27]
df.loc[(df['Duration of Credit (month)'] >= 24) & (df['Duration of Credit (month)'] <= 27), 'Duration of Credit (month)'] = 3
# de (28 a 75]
df.loc[(df['Duration of Credit (month)'] >= 28) & (df['Duration of Credit (month)'] <= 75), 'Duration of Credit (month)'] = 4

# de [0 a 26]
df.loc[(df['Age (years)'] >= 0) & (df['Age (years)'] <= 26), 'Age (years)'] = 0
# de [27 a 32]
df.loc[(df['Age (years)'] >= 27) & (df['Age (years)'] <= 32), 'Age (years)'] = 1
# de [33 a 42]
df.loc[(df['Age (years)'] >= 33) & (df['Age (years)'] <= 42), 'Age (years)'] = 2
# de [43 a 75]
df.loc[(df['Age (years)'] >= 43) & (df['Age (years)'] <= 75), 'Age (years)'] = 3


# de [0 a 1500)
df.loc[(df['Credit Amount'] >= 0) & (df['Credit Amount'] < 1500), 'Credit Amount'] = 0
# de [1500 a 3000)
df.loc[(df['Credit Amount'] >= 1500) & (df['Credit Amount'] < 3000), 'Credit Amount'] = 1
# de [3000 a inf)
df.loc[df['Credit Amount'] >= 3000, 'Credit Amount'] = 2


df.to_csv('german_credit_adjusted.csv', index=False)



