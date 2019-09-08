# Original Code : https://github.com/alrojo/CB513/blob/master/data.py

import os
import numpy as np
import pandas as pd
import subprocess
from utils import load_gz


#TRAIN_PATH = '../pssp-data/cullpdb+profile_6133_filtered.npy.gz'
#TEST_PATH = '../pssp-data/cb513+profile_split1.npy.gz'
TRAIN_PATH = '../pssp-data/aa_train.txt'
TEST_PATH = '../pssp-data/aa_test.txt'
TRAIN_DATASET_PATH = '../pssp-data/train.npz'
TEST_DATASET_PATH = '../pssp-data/test.npz'
TRAIN_URL = 'http://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_6133_filtered.npy.gz'
TEST_URL = 'http://www.princeton.edu/~jzthree/datasets/ICML2014/cb513+profile_split1.npy.gz'

N_STATE = 3

def download_dataset():
    print('[Info] Downloading CB513 dataset ...')
    if not (os.path.isfile(TRAIN_PATH) and os.path.isfile(TEST_PATH)):
        os.makedirs('../pssp-data', exist_ok=True)
        os.system(f'wget -O {TRAIN_PATH} {TRAIN_URL}')
        os.system(f'wget -O {TEST_PATH} {TEST_URL}')


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

    #data = load_gz(path)
    data = pd.read_csv(path, header=None, sep=' ')

    oe = OneHotEncoder(categories='auto', sparse=False)
    X = oe.fit_transform(data)

    X = np.resize(X, (data.shape[0], 23, 20))
    X = X.transpose(0, 2, 1)
    X = X.astype('float32')
    print(X.shape)

    #data = data.reshape(-1, 700, 57)

    data = pd.read_csv(path.replace('aa', 'pss'), header=None, sep=' ')

    le = LabelEncoder()
    le.fit(data[0])
    for col in data.columns:
        data[col] = le.transform(data[col])
    y = data.astype('float32')
    print(y.shape)

    print(data.shape[0])
    seq_len = [23] * data.shape[0]
    seq_len = np.asarray(seq_len).astype('float32')

    #idx = np.append(np.arange(21), np.arange(35, 56))
    '''
    idx = np.arange(21)
    X = data[:, :, idx]
    X = X.transpose(0, 2, 1)
    X = X.astype('float32')

    y = data[:, :, 22:30]
    y = np.array([np.dot(yi, np.arange(N_STATE)) for yi in y])
    y = y.astype('float32')

    mask = data[:, :, 30] * -1 + 1
    seq_len = mask.sum(axis=1)
    seq_len = seq_len.astype('float32')
    '''

    return X, y, seq_len
