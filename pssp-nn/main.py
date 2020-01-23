import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
import argparse
import json
import os

N_STATE = 3
N_AA = 20
N_LEN = 23

TRAIN_PATH = '../../helix/csv/aa_train.txt'
TEST_PATH = '../../helix/csv/aa_test.txt'

class CrossEntropy(object):
    def __init__(self):
        pass

    def __call__(self, out, target, seq_len):
        loss = 0
        for o, t, l in zip(out, target, seq_len):
            loss += nn.CrossEntropyLoss()(o[:l], t[:l])
        return loss

# class LossFunc(object):
#     def __init__(self):
#         self.loss = nn.CrossEntropyLoss()
#
#     def __call__(self, out, target, seq_len):
#         '''
#         out.shape : (batch_size, class_num, seq_len)
#         target.shape : (batch_size, seq_len)
#         '''
#         out = torch.clamp(out, 1e-15, 1 - 1e-15)
#         return torch.tensor([self.loss(o[:l], t[:l])
#                              for o, t, l in zip(out, target, seq_len)],
#                             requires_grad=True).sum()

def args2json(data, path, print_args=True):
    data = vars(data)
    if print_args:
        print(f'\n+ ---------------------------')
        for k, v in data.items():
            print(f'  {k.upper()} : {v}')
        print(f'+ ---------------------------\n')

    with open(os.path.join(path, 'args.json'), 'w') as f:
        json.dump(data, f)

def accuracy(out, target, seq_len):
    '''
    out.shape : (batch_size, seq_len, class_num)
    target.shape : (class_num, seq_len)
    seq_len.shape : (batch_size)
    '''
    out = out.cpu().data.numpy()
    target = target.cpu().data.numpy()
    seq_len = seq_len.cpu().data.numpy()

    out = out.argmax(axis=2)
    return np.array([np.equal(o[:l], t[:l]).sum()/l for o, t, l in zip(out, target, seq_len)]).mean()

def show_progress(e, e_total, train_loss, test_loss, train_acc, acc):
    print(f'[{e:3d}/{e_total:3d}] train_loss:{train_loss:.2f}, '\
        f'test_loss:{test_loss:.2f}, train_acc:{train_acc:.3f} acc:{acc:.3f}')

def save_history(history, save_dir):
    save_path = os.path.join(save_dir, 'history.npy')
    np.save(save_path, history)

def save_model(model, save_dir):
    save_path = os.path.join(save_dir, 'model.pth')
    torch.save(model.state_dict(), save_path)

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
    y = data.astype('float32').values

    seq_len = [N_LEN] * data.shape[0]
    seq_len = np.asarray(seq_len).astype('float32')

    return X, y, seq_len

class MyDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y.astype(int)
        self.seq_len = seq_len.astype(int)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        seq_len = self.seq_len[idx]
        return x, y, seq_len

class LoadDataset(object):
    def __init__(self, batch_size_train, batch_size_test):
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test

    def load_dataset(self):
        # train dataset
        X_train, y_train, seq_len_train = make_dataset(TRAIN_PATH)

        # test dataset
        X_test, y_test, seq_len_test = make_dataset(TEST_PATH)
        
        return X_train, y_train, seq_len_train, X_test, y_test, seq_len_test

    def __call__(self):
        X_train, y_train, seq_len_train, X_test, y_test, seq_len_test = self.load_dataset()

        D_train = MyDataset(X_train, y_train, seq_len_train)
        train_loader = DataLoader(D_train, batch_size=self.batch_size_train, shuffle=True)

        D_test = MyDataset(X_test, y_test, seq_len_test)
        test_loader = DataLoader(D_test, batch_size=self.batch_size_test, shuffle=False)

        return train_loader, test_loader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        conv_hidden_size = 64
        self.conv1 = nn.Sequential(
            nn.Conv1d(N_AA, conv_hidden_size, 3, 1, 3 // 2),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv1d(N_AA, conv_hidden_size, 7, 1, 7 // 2),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv1d(N_AA, conv_hidden_size, 11, 1, 11 // 2),
            nn.ReLU())

        # LSTM(input_size, hidden_size, num_layers, bias,
        #      batch_first, dropout, bidirectional)
        rnn_hidden_size = 256
        self.brnn = nn.GRU(conv_hidden_size*3, rnn_hidden_size, 3, True, True, 0.5, True)

        self.fc = nn.Sequential(
                nn.Linear(rnn_hidden_size*2+conv_hidden_size*3, 126),
                nn.ReLU(),
                nn.Linear(126, N_STATE),
                nn.ReLU())

    def forward(self, x):
        # obtain multiple local contextual feature map
        conv_out = torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], dim=1)

        # Turn (batch_size x hidden_size x seq_len)
        # into (batch_size x seq_len x hidden_size)
        conv_out = conv_out.transpose(1, 2)

        # bidirectional rnn
        out, _ = self.brnn(conv_out)

        out = torch.cat([conv_out, out], dim=2)
        # print(out.sum())

        # Output shape is (batch_size x seq_len x classnum)
        out = self.fc(out)
        out = F.softmax(out, dim=2)
        return out

def train(model, device, train_loader, optimizer, loss_function):
    model.train()
    train_loss = 0
    acc = 0
    len_ = len(train_loader)
    for batch_idx, (data, target, seq_len) in enumerate(train_loader):
        data, target, seq_len = data.to(device), target.to(device), seq_len.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_function(out, target, seq_len)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        acc += accuracy(out, target, seq_len)

    train_loss /= len_
    acc /= len_
    return train_loss, acc

def test(model, device, test_loader, loss_function):
    model.eval()
    test_loss = 0
    acc = 0
    len_ = len(test_loader)
    with torch.no_grad():
        for i, (data, target, seq_len) in enumerate(test_loader):
            data, target, seq_len = data.to(device), target.to(device), seq_len.to(device)
            out = model(data)
            test_loss += loss_function(out, target, seq_len).cpu().data.numpy()
            acc += accuracy(out, target, seq_len)

    test_loss /= len_
    acc /= len_
    return test_loss, acc

def main():
    # params
    # ----------
    parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='The number of epochs to run (default: 100)')
    parser.add_argument('-b', '--batch_size_train', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('-b_test', '--batch_size_test', type=int, default=1024,
                        help='input batch size for testing (default: 1024)')
    parser.add_argument('--result_dir', type=str, default='./result',
                        help='Output directory (default: ./result)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enable CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if use_cuda else 'cpu')

    # make directory to save train history and model
    os.makedirs(args.result_dir, exist_ok=True)
    args2json(args, args.result_dir)

    # laod dataset and set k-fold cross validation
    D = LoadDataset(args.batch_size_train, args.batch_size_test)
    train_loader, test_loader = D()

    # model, loss_function, optimizer
    model = Net().to(device)
    
    if use_cuda:
        model = torch.nn.DataParallel(model)

    loss_function = CrossEntropy()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01)

    # train and test
    history = []
    for e in range(args.epochs):
        train_loss, train_acc = train(model, device, train_loader, optimizer, loss_function)
        test_loss, acc = test(model, device, test_loader, loss_function)
        history.append([train_loss, test_loss, train_acc, acc])
        show_progress(e+1, args.epochs, train_loss, test_loss, train_acc, acc)

    # save train history and model
    save_history(history, args.result_dir)
    save_model(model, args.result_dir)

if __name__ == '__main__':
    main()
