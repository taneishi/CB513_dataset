import numpy as np
import argparse
import timeit
import os

import torch
import torch.nn as nn
import torch.utils.data

from models import ConvNet
from sequences import data_load

class CrossEntropy(object):
    def __init__(self):
        self.loss_function = nn.CrossEntropyLoss()

    def __call__(self, out, target, seq_len):
        loss = sum(self.loss_function(o[:l], t[:l]) for o, t, l in zip(out, target, seq_len))
        return loss

def accuracy(out, target, seq_len):
    '''
    out.shape : (batch_size, seq_len, class_num)
    target.shape : (class_num, seq_len)
    '''
    out = out.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    seq_len = seq_len.cpu().detach().numpy()
    out = out.argmax(axis=2)

    return np.array([np.equal(o[:l], t[:l]).sum()/l for o, t, l in zip(out, target, seq_len)]).mean()

def main(args):
    torch.manual_seed(args.random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {} device.'.format(device))

    X, y, seq_len = data_load(args.train_path)
    X = torch.FloatTensor(X).to(device)
    y = torch.LongTensor(y).to(device)
    seq_len = torch.ShortTensor(seq_len).to(device)
    train_dataset = torch.utils.data.TensorDataset(X, y, seq_len)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True)
    print('Training set {} sequences {} batches.'.format(len(train_dataset), len(train_loader)))

    X, y, seq_len = data_load(args.test_path)
    X = torch.FloatTensor(X).to(device)
    y = torch.LongTensor(y).to(device)
    seq_len = torch.ShortTensor(seq_len).to(device)
    test_dataset = torch.utils.data.TensorDataset(X, y, seq_len)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_test)
    print('Test set {} sequences {} batches.'.format(len(test_dataset), len(test_loader)))

    model = ConvNet().to(device)

    if args.model_path and os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path))

    loss_function = CrossEntropy()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        epoch_start = timeit.default_timer()

        model.train()
        train_loss = 0
        train_acc = 0
        for index, (data, target, seq_len) in enumerate(train_loader, 1):
            out = model(data)
            loss = loss_function(out, target, seq_len)
            train_loss += loss.item() / len(data)
            train_acc += accuracy(out, target, seq_len)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('\repoch {:3d}/{:3d} batch {:3d}/{:3d}'.format(epoch + 1, args.epochs, index, len(train_loader)), end='')
            print(' train_loss {:5.3f} train_acc {:5.3f}'.format(train_loss / index, train_acc / index)), end='')

        model.eval()
        test_loss = 0
        test_acc = 0
        for data, target, seq_len in test_loader:
            with torch.no_grad():
                out = model(data)
            loss = loss_function(out, target, seq_len)
            test_loss += loss.item() / len(data)
            test_acc += accuracy(out, target, seq_len)

        print(' test_loss {:5.3f} test_acc {:5.3f}'.format(test_loss / len(test_loader), test_acc / len(test_loader)), end='')
        print(' {:5.2f}sec'.format(timeit.default_timer() - epoch_start))

        torch.save(model.state_dict(), args.model_path)

    print('')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')
    parser.add_argument('--random_seed', default=123, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--train_path', default='cullpdb+profile_5926_filtered.npy.gz')
    parser.add_argument('--test_path', default='cb513+profile_split1.npy.gz')
    parser.add_argument('--batch_size_train', default=100, type=int, help='input batch size for training (default: 100)')
    parser.add_argument('--batch_size_test', default=1000, type=int, help='input batch size for testing (default: 1000)')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--model_path', default='model/checkpoint.pth', type=str)
    args = parser.parse_args()
    print(vars(args))

    main(args)
