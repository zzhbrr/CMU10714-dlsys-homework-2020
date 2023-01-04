import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)

class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, device=None, dtype="float32"):
        super().__init__()
        self.conv = nn.Conv(in_channels, out_channels, kernel_size, stride, bias=True, device=device, dtype=dtype)
        self.bn = nn.BatchNorm2d(dim=out_channels, device=device)
        self.relu = nn.ReLU()
    def forward(self, x):
        ret = self.conv(x)
        ret = self.bn(ret)
        ret = self.relu(ret)
        return ret


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.blocks = []
        self.blocks.append(ConvBN(3, 16, 7, 4, device=device, dtype=dtype))
        self.blocks.append(ConvBN(16, 32, 3, 2, device=device, dtype=dtype))
        self.blocks.append(nn.Residual(nn.Sequential(ConvBN(32, 32, 3, 1, device=device, dtype=dtype), \
                        ConvBN(32, 32, 3, 1, device=device, dtype=dtype))))
        self.blocks.append(ConvBN(32, 64, 3, 2, device=device, dtype=dtype))
        self.blocks.append(ConvBN(64, 128, 3, 2, device=device, dtype=dtype))
        self.blocks.append(nn.Residual(nn.Sequential(ConvBN(128, 128, 3, 1, device=device, dtype=dtype), \
                        ConvBN(128, 128, 3, 1, device=device, dtype=dtype))))
        self.blocks.append(nn.Flatten())
        self.blocks.append(nn.Linear(128, 128, device=device, dtype=dtype))
        self.blocks.append(nn.ReLU())
        self.blocks.append(nn.Linear(128, 10, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        res = x
        for block in self.blocks:
            res = block(res)
        return res
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        self.seq_model = seq_model
        if seq_model == 'rnn':
            self.model = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        else:
            self.model = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        self.Linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        embed_x = self.embedding(x) # shape of (seq_len, bs, embedding_size)
        out_features, h = self.model(embed_x, h) # out_features shape of (seq_len, bs, hidden_size)
        out = self.Linear(out_features.reshape((out_features.shape[0]*out_features.shape[1], out_features.shape[2])))
        return out, h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)