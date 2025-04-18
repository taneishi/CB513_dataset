import torch.nn.functional as F
import torch.nn as nn
import torch

class ConvNet(nn.Module):
    def __init__(self, n_aa=21, n_state=8):
        super(ConvNet, self).__init__()

        # Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        conv_hidden_size = 64

        self.conv1 = nn.Sequential(nn.Conv1d(n_aa, conv_hidden_size, 3, 1, 3 // 2), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(n_aa, conv_hidden_size, 7, 1, 7 // 2), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(n_aa, conv_hidden_size, 11, 1, 11 // 2), nn.ReLU())

        # LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
        rnn_hidden_size = 256

        self.brnn = nn.GRU(conv_hidden_size*3, rnn_hidden_size, 3, True, True, 0.5, True)

        self.fc = nn.Sequential(
                nn.Linear(rnn_hidden_size*2+conv_hidden_size*3, 128),
                nn.ReLU(),
                nn.Linear(128, n_state),
                nn.ReLU())

    def forward(self, x):
        # Obtain multiple local contextual feature map.
        conv_out = torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], dim=1)

        # Turn (batch_size x hidden_size x seq_len) into (batch_size x seq_len x hidden_size).
        conv_out = conv_out.transpose(1, 2)

        # Bidirectional RNN.
        out, _ = self.brnn(conv_out)

        out = torch.cat([conv_out, out], dim=2)
        # print(out.sum())

        # Output shape is (batch_size x seq_len x classnum).
        out = self.fc(out)
        out = F.softmax(out, dim=2)
        return out
