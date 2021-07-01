#!/bin/env /bin/python
"""Split and prepare files from Kaggle.com for loading into Tensorflow"""

import pandas as pd
import numpy as np

df_true = pd.read_csv('./archive/True.csv')
df_false = pd.read_csv('./archive/Fake.csv')

df_true.loc[:,'bool'] = 1
df_false.loc[:,'bool'] = 0
df = df_true.append(df_false)
df = df.sample(frac = 1)

TEST_SIZE= 0.20

def split_set(dataframe, test_size):
    """Returns two dataframes of indicated size"""
    i = np.floor(len(dataframe)*test_size).astype(int)
    set_a = dataframe[0:i].reset_index()
    set_b = dataframe[i:].reset_index()
    return set_a, set_b

train, test = split_set(df, TEST_SIZE)

for ind in train.index:
    if train.loc[ind,'bool'] == 1:
        FILENAME = './input/train/true/true_text_' + str(ind+1) + '.txt'
    else:
        FILENAME = './input/train/false/false_text_' + str(ind+1) + '.txt'
    text = train.iloc[ind,2].lower()
    with open(FILENAME,'w') as f:
        f.write(text)


for ind in test.index:
    if test.loc[ind,'bool'] == 1:
        FILENAME = './input/test/true/true_text_' + str(ind+1) + '.txt'
    else:
        FILENAME = './input/test/false/false_text_' + str(ind+1) + '.txt'
    text = test.iloc[ind,2].lower()
    with open(FILENAME,'w') as f:
        f.write(text)
