import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

#!wget -c -P data http://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_5926_filtered.npy.gz
#!wget -c -P data http://www.princeton.edu/~jzthree/datasets/ICML2014/cb513+profile_split1.npy.gz

AA = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X', 'NoSeq']
SS = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T', 'NoSeq']

def plot_secondary_structure(train_ss, test_ss):
    plt.figure(figsize=(8, 3))

    plt.subplot(1, 2, 1)
    plt.title('Training set')
    plt.bar(train_ss.index, train_ss['count'], width=0.5)
    plt.grid(True, axis='y')

    plt.subplot(1, 2, 2)
    plt.title('Test set')
    plt.bar(test_ss.index, test_ss['count'], width=0.5)
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('figure/ss.png')

def plot_amino_acid(train_aa, test_aa):
    plt.figure(figsize=(8, 3))

    plt.subplot(1, 2, 1)
    plt.title('Training set')
    plt.bar(train_aa.index, train_aa['count'], width=0.6)
    plt.grid(True, axis='y')

    plt.subplot(1, 2, 2)
    plt.title('Test set')
    plt.bar(test_aa.index, test_aa['count'], width=0.6)
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('figure/amino_acid.png', dpi=100)

def plot_sequence_length(train_seq_len, test_seq_len):
    plt.figure(figsize=(8, 3))

    plt.subplot(1, 2, 1)
    plt.title('Training set')
    plt.hist(train_seq_len, bins=100)
    plt.grid(True, axis='y')

    plt.subplot(1, 2, 2)
    plt.title('Test set')
    plt.hist(test_seq_len, bins=100)
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('figure/seq_len.png', dpi=100)

def data_load(path):
    data = np.load('data/{}'.format(path))

    # Original 57 features.
    data = data.reshape(-1, 700, 57)

    # 20-residues + non-seq.
    X = data[:, :, np.arange(21)]
    X = X.transpose(0, 2, 1)
    X = X.astype('float32')

    # 8-states.
    y = data[:, :, 22:30]
    y = np.array([np.dot(yi, np.arange(8)) for yi in y])
    y = y.astype('float32')

    mask = data[:, :, 30] * -1 + 1
    seq_len = mask.sum(axis=1)
    seq_len = seq_len.astype('int')

    return X, y, seq_len

def main(args):
    train_X, train_y, train_seq_len = data_load(args.train_path)
    test_X, test_y, test_seq_len = data_load(args.test_path)

    # Sequence length.
    print('Training set {} sequences.'.format(len(train_seq_len)))
    print('Test set {} sequences.'.format(len(test_seq_len)))

    plot_sequence_length(train_seq_len, test_seq_len)

    # Amino acid residues.
    train_aa = []
    train_seq_aa = []
    for seq, seq_len in zip(train_X, train_seq_len):
        seq_aa = []
        for aa in seq[:, :seq_len].T:
            assert len(np.where(aa == 1)) == 1
            seq_aa.append(AA[aa.argmax()])
        train_aa += seq_aa
        train_seq_aa.append(''.join(seq_aa))

    train_aa = pd.DataFrame(train_aa, columns=['AA'])
    train_aa = train_aa.groupby('AA').aggregate(count=('AA', 'size'))

    pd.DataFrame(train_seq_aa, columns=['AA']).to_csv('data/train.aa', index=False)

    test_aa = []
    test_seq_aa = []
    for seq, seq_len in zip(test_X, test_seq_len):
        seq_aa = []
        for aa in seq[:, :seq_len].T:
            assert len(np.where(aa == 1)) == 1
            seq_aa.append(AA[aa.argmax()])
        test_aa += seq_aa
        test_seq_aa.append(''.join(seq_aa))

    test_aa = pd.DataFrame(test_aa, columns=['AA'])
    test_aa = test_aa.groupby('AA').aggregate(count=('AA', 'size'))

    pd.DataFrame(test_seq_aa, columns=['AA']).to_csv('data/test.aa', index=False)

    plot_amino_acid(train_aa, test_aa)

    # 8-state secondary structures.
    train_ss = []
    train_seq_ss = []
    for seq, seq_len in zip(train_y, train_seq_len):
        seq_ss = []
        for ss in seq[:seq_len].astype(int):
            seq_ss.append(SS[ss])
        train_ss += seq_ss
        train_seq_ss.append(''.join(seq_ss))

    train_ss = pd.DataFrame(train_ss, columns=['SS'])
    train_ss = train_ss.groupby('SS').aggregate(count=('SS', 'size'))

    pd.DataFrame(train_seq_ss, columns=['SS']).to_csv('data/train.ss', index=False)

    test_ss = []
    test_seq_ss = []
    for seq, seq_len in zip(test_y, test_seq_len):
        seq_ss = []
        for ss in seq[:seq_len].astype(int):
            seq_ss.append(SS[ss])
        test_ss += seq_ss
        test_seq_ss.append(''.join(seq_ss))

    test_ss = pd.DataFrame(test_ss, columns=['SS'])
    test_ss = test_ss.groupby('SS').aggregate(count=('SS', 'size'))

    pd.DataFrame(test_seq_ss, columns=['SS']).to_csv('data/test.ss', index=False)

    plot_secondary_structure(train_ss, test_ss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')
    parser.add_argument('--train_path', default='cullpdb+profile_5926_filtered.npy.gz')
    parser.add_argument('--test_path', default='cb513+profile_split1.npy.gz')
    args = parser.parse_args()
    print(vars(args))

    main(args)
