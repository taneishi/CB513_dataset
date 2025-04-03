import pandas as pd
import numpy as np
import argparse

AA = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X', 'NoSeq']
SS = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T', 'NoSeq']

def data_load(path):
    data = np.load(f'data/{path}')

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
    print(f'Training set {len(train_seq_len)} sequences.')
    print(f'Test set {len(test_seq_len)} sequences.')

    # Amino acid residues.
    train_aa = []
    for seq, seq_len in zip(train_X, train_seq_len):
        seq_aa = []
        for aa in seq[:, :seq_len].T:
            assert len(np.where(aa == 1)) == 1
            seq_aa.append(AA[aa.argmax()])
        train_aa.append(''.join(seq_aa))
    pd.DataFrame(train_aa, columns=['AA']).to_csv('data/train.aa', index=False)

    test_aa = []
    for seq, seq_len in zip(test_X, test_seq_len):
        seq_aa = []
        for aa in seq[:, :seq_len].T:
            assert len(np.where(aa == 1)) == 1
            seq_aa.append(AA[aa.argmax()])
        test_aa.append(''.join(seq_aa))
    pd.DataFrame(test_aa, columns=['AA']).to_csv('data/test.aa', index=False)

    # Secondary structures.
    train_ss = []
    for seq, seq_len in zip(train_y, train_seq_len):
        seq_ss = []
        for ss in seq[:seq_len].astype(int):
            seq_ss.append(SS[ss])
        train_ss.append(''.join(seq_ss))
    pd.DataFrame(train_ss, columns=['SS']).to_csv('data/train.ss', index=False)

    test_ss = []
    for seq, seq_len in zip(test_y, test_seq_len):
        seq_ss = []
        for ss in seq[:seq_len].astype(int):
            seq_ss.append(SS[ss])
        test_ss.append(''.join(seq_ss))
    pd.DataFrame(test_ss, columns=['SS']).to_csv('data/test.ss', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')
    parser.add_argument('--train_path', default='cullpdb+profile_5926_filtered.npy.gz')
    parser.add_argument('--test_path', default='cb513+profile_split1.npy.gz')
    args = parser.parse_args()
    print(vars(args))

    main(args)
