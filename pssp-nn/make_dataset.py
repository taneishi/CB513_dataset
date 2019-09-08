# Original Code : https://github.com/alrojo/CB513/blob/master/data.py

import os
import numpy as np
import pandas as pd
import subprocess
from utils import load_gz


TRAIN_PATH = '../pssp-data/aa_train.txt'
TEST_PATH = '../pssp-data/aa_test.txt'
TRAIN_DATASET_PATH = '../pssp-data/train.npz'
TEST_DATASET_PATH = '../pssp-data/test.npz'

N_STATE = 3
N_LEN = 23
N_AA = 20


def make_datasets():
    print('[Info] Making datasets ...')

    # train dataset
    X_train, y_train, seq_len_train = make_dataset(TRAIN_PATH)
    np.savez_compressed(TRAIN_DATASET_PATH, X=X_train, y=y_train, seq_len=seq_len_train)
    print(f'[Info] Saved train dataset in {TRAIN_DATASET_PATH}')

    # test dataset
    X_test, y_test, seq_len_test = make_dataset(TEST_PATH)
    np.savez_compressed(TEST_DATASET_PATH, X=X_test, y=y_test, seq_len=seq_len_test)
    print(f'[Info] Saved test dataset in {TEST_DATASET_PATH}')


def make_dataset(path):
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder

    data = pd.read_csv(path, header=None, sep=' ')

    oe = OneHotEncoder(categories='auto', sparse=False)
    X = oe.fit_transform(data)

    X = np.resize(X, (data.shape[0], N_LEN, N_AA))
    X = X.transpose(0, 2, 1)
    X = X.astype('float32')

    data = pd.read_csv(path.replace('aa', 'pss'), header=None, sep=' ')

    le = LabelEncoder()
    le.fit(data[0])
    for col in data.columns:
        data[col] = le.transform(data[col])
    y = data.astype('float32')

    seq_len = [N_LEN] * data.shape[0]
    seq_len = np.asarray(seq_len).astype('float32')

    return X, y, seq_len
