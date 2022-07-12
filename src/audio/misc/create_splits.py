from sklearn.model_selection import train_test_split
import pandas as pd
import argparse, os

import sys
sys.path.append('src/audio/constants/')
from misc import RANDOM_STATE

DATASET = 'assets/audio/data/Dataset/'
SPLIT_BASE_PATH = 'assets/audio/dataset_split/'

TRAIN_NAME = 'train_split.txt'
VALIDATION_NAME = 'val_split.txt'
TEST_NAME = 'test_split.txt'

fnames, classes = [], []

for folder in ['Bark', 'Doorbell', 'Drill', 'Hammer', 'Glass', 'Speech']:
    for file in os.listdir(os.path.join(DATASET,folder)):
        if file.split(".")[-1] == 'wav':
          classes.append(folder)
          fnames.append(file)

df = pd.DataFrame({'fname':fnames, 'label':classes})

y = df.label
X = df.drop(columns=['label'])


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.13, stratify=y_train, random_state=RANDOM_STATE) 

with open(os.path.join(SPLIT_BASE_PATH, TRAIN_NAME), 'w') as file:
    for idx in range(len(x_train)):
        string = './{}{}/{}\n'.format(DATASET, y_train.iloc[idx], x_train.iloc[idx].fname)
        file.write(string)
file.close()

with open(os.path.join(SPLIT_BASE_PATH, VALIDATION_NAME), 'w') as file:
    for idx in range(len(x_val)):
        string = './{}{}/{}\n'.format(DATASET, y_val.iloc[idx], x_val.iloc[idx].fname)
        file.write(string)
file.close()

with open(os.path.join(SPLIT_BASE_PATH, TEST_NAME), 'w') as file:
    for idx in range(len(x_test)):
        string = './{}{}/{}\n'.format(DATASET, y_test.iloc[idx], x_test.iloc[idx].fname)
        file.write(string)
file.close()
