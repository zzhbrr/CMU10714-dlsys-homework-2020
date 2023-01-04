"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, requires_grad=True, device=device))
        if bias:
            self.bias = Parameter(ops.reshape(init.kaiming_uniform(out_features, 1, requires_grad=True, device=device), (1, out_features)))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mul = ops.matmul(X, self.weight)
        if self.bias is None:
            return mul
        else:
            return mul + ops.broadcast_to(self.bias, mul.shape)
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        totsz = 1
        for shp in X.shape:
            totsz *= shp
        return ops.reshape(X, (X.shape[0], int(totsz/X.shape[0])))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (1+ops.exp(-x))**(-1)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        res = x
        for module in self.modules:
            res = module(res)
        return res
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        exp_sum = ops.logsumexp(logits, axes=(1, )).sum()
        z_y_sum = (logits * init.one_hot(logits.shape[1], y, logits.device, logits.dtype)).sum()
        return (exp_sum - z_y_sum) / logits.shape[0]

        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim, requires_grad=True, device=device))
        self.bias = Parameter(init.zeros(self.dim, requires_grad=True, device=device))
        self.running_mean = init.zeros(self.dim, requires_grad=False, device=device)
        self.running_var = init.ones(self.dim, requires_grad=False, device=device)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            E = ops.summation(x, axes=(0,))/ x.shape[0]
            print('BatchNorm1d: E shape: ', E.shape)
            print('BatchNorm1d: x shape: ', x.shape)
            E_bdct = ops.broadcast_to(E, x.shape)
            var = ops.summation((x - E_bdct)**2, axes=(0,))/x.shape[0]
            fm = ops.broadcast_to((var + self.eps)**0.5, x.shape)
            y = ops.broadcast_to(self.weight, x.shape) * (x - E_bdct) / fm + ops.broadcast_to(self.bias, x.shape)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * E.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data
        else:
            fm = ops.broadcast_to((self.running_var + self.eps)**0.5, x.shape)
            fz = x - ops.broadcast_to(self.running_mean, x.shape)
            y = ops.broadcast_to(self.weight, x.shape) * (fz / fm) + ops.broadcast_to(self.bias, x.shape)
        return y
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(*(1, dim), requires_grad=True))
        self.bias = Parameter(init.zeros(*(1, dim), requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        E = ops.broadcast_to(ops.reshape(ops.summation(x, axes=(1,)), (x.shape[0], 1)) / self.dim, x.shape)
        fm = ops.broadcast_to(ops.reshape((ops.summation((x - E)**2, axes=(1,))/self.dim + self.eps)**0.5, (x.shape[0],1)), x.shape)
        y = ops.broadcast_to(self.weight, x.shape) * (x - E) / fm + ops.broadcast_to(self.bias, x.shape)
        return y
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            return x / (1 - self.p) * init.randb(*x.shape, p=1-self.p)
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_channels*kernel_size*kernel_size, \
            out_channels*kernel_size*kernel_size, \
            (kernel_size, kernel_size, in_channels, out_channels), \
            device=device, requires_grad=True))
        if bias:
            self.bias = Parameter(init.rand(*(out_channels, ), \
                low=-1.0/(in_channels*kernel_size**2)**0.5, \
                high=1.0/(in_channels*kernel_size**2)**0.5, \
                device=device, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N, C, H, W = x.shape
        x = (x.transpose((1, 2))).transpose((2, 3))
        padding = int((self.kernel_size - 1) / 2)
        res = ops.conv(x, self.weight, stride=self.stride, padding=padding)
        if self.bias:
            res = res + ops.broadcast_to(ops.reshape(self.bias, (1,1,1,self.out_channels)), res.shape)
        return ops.transpose(ops.transpose(res, (2,3)), (1,2))
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.W_ih = Parameter(init.rand(*(input_size, hidden_size), low = -(1/hidden_size)**0.5, \
            high = (1/hidden_size)**0.5, device=device, requires_grad=True))
        self.W_hh = Parameter(init.rand(*(hidden_size, hidden_size), low = -(1/hidden_size)**0.5, \
            high = (1/hidden_size)**0.5, device=device, requires_grad=True))
        if bias:
            self.bias_ih = Parameter(init.rand(*(hidden_size,), low = -(1/hidden_size)**0.5, \
                high = (1/hidden_size)**0.5, device=device, requires_grad=True))
            self.bias_hh = Parameter(init.rand(*(hidden_size,), low = -(1/hidden_size)**0.5, \
                high = (1/hidden_size)**0.5, device=device, requires_grad=True))
        else:
            self.bias_ih = None
            self.bias_hh = None
        if nonlinearity == 'tanh':
            self.activ = Tanh()
        else: 
            self.activ = ReLU()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = init.zeros(*(X.shape[0], self.hidden_size), device=X.device)
        ret = ops.matmul(X, self.W_ih) + ops.matmul(h, self.W_hh)
        if self.bias_ih:
            ret = ret + ops.broadcast_to(self.bias_ih.reshape((1, self.hidden_size)), ret.shape)
        if self.bias_hh:
            ret = ret + ops.broadcast_to(self.bias_hh.reshape((1, self.hidden_size)), ret.shape)
        return self.activ(ret)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn_cells = []
        for i in range(num_layers):
            if i == 0:
                self.rnn_cells.append(RNNCell(input_size, hidden_size, bias=bias, nonlinearity=nonlinearity, device=device, dtype=dtype))
            else:
                self.rnn_cells.append(RNNCell(hidden_size, hidden_size, bias=bias, nonlinearity=nonlinearity, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape
        if h0 is None:
            h0 = init.zeros(*(self.num_layers, bs, self.hidden_size), device=X.device)
        
        out_features = init.zeros(*(seq_len, bs, self.hidden_size), device=X.device, requires_grad=True)
        out_h = init.zeros(*(self.num_layers, bs, self.hidden_size), device=X.device, requires_grad=True)
        
        out_features_split = list(ops.split(out_features, 0))
        out_h_split = list(ops.split(out_h, 0))
        X_split = list(ops.split(X, 0))
        h0_split = list(ops.split(h0, 0))

        for i in range(self.num_layers):
            h = h0_split[i]
            for j in range(seq_len):
                h = self.rnn_cells[i](X_split[j], h)
                X_split[j] = h
                if i == self.num_layers - 1:
                    out_features_split[j] = h
            out_h_split[i] = h
        out_features = ops.stack(tuple(out_features_split), 0)
        out_h = ops.stack(tuple(out_h_split), 0)
        return out_features, out_h
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.W_ih = Parameter(init.rand(*(input_size, 4*hidden_size), low = -(1/hidden_size)**0.5, \
            high = (1/hidden_size)**0.5, device=device, requires_grad=True))
        self.W_hh = Parameter(init.rand(*(hidden_size, 4*hidden_size), low = -(1/hidden_size)**0.5, \
            high = (1/hidden_size)**0.5, device=device, requires_grad=True))
        if bias:
            self.bias_ih = Parameter(init.rand(*(4*hidden_size,), low = -(1/hidden_size)**0.5, \
                high = (1/hidden_size)**0.5, device=device, requires_grad=True))
            self.bias_hh = Parameter(init.rand(*(4*hidden_size,), low = -(1/hidden_size)**0.5, \
                high = (1/hidden_size)**0.5, device=device, requires_grad=True))
        else:
            self.bias_ih = None
            self.bias_hh = None
        self.bias = bias
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs, input_size = X.shape
        if h is None:
            h = (init.zeros(*(bs, self.hidden_size), device=X.device), init.zeros(*(bs, self.hidden_size), device=X.device))
        h0, c0 = h
        res = X @ self.W_ih + h0 @ self.W_hh
        if self.bias_ih:
            res = res + ops.broadcast_to(self.bias_ih, res.shape)
        if self.bias_hh:
            res = res + ops.broadcast_to(self.bias_hh, res.shape)
        res = ops.reshape(res, (bs, 4, self.hidden_size))
        res = ops.transpose(res, (0, 1))
        res = ops.split(res, 0)
        i = self.sigmoid(res[0])
        f = self.sigmoid(res[1])
        g = self.tanh(res[2])
        o = self.sigmoid(res[3])
        c = f * c0 + i * g
        h = o * self.tanh(c)
        return h, c
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm_cells = []
        for i in range(num_layers):
            if i == 0:
                self.lstm_cells.append(LSTMCell(input_size, hidden_size, bias=bias, device=device, dtype=dtype))
            else:
                self.lstm_cells.append(LSTMCell(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape
        if h is None:
            h = (init.zeros(*(self.num_layers, bs, self.hidden_size), device=X.device), init.zeros(*(self.num_layers, bs, self.hidden_size), device=X.device))
        h0, c0 = h

        out_features = init.zeros(*(seq_len, bs, self.hidden_size), device=X.device, requires_grad=True)
        out_h = init.zeros(*(self.num_layers, bs, self.hidden_size), device=X.device, requires_grad=True)
        out_c = init.zeros(*(self.num_layers, bs, self.hidden_size), device=X.device, requires_grad=True)

        out_features_split = list(ops.split(out_features, 0))
        out_h_split = list(ops.split(out_h, 0))
        out_c_split = list(ops.split(out_c, 0))
        X_split = list(ops.split(X, 0))
        h0_split = list(ops.split(h0, 0))
        c0_split = list(ops.split(c0, 0))

        for i in range(self.num_layers):
            h_t = h0_split[i]
            c_t = c0_split[i]
            for j in range(seq_len):
                h_t, c_t = self.lstm_cells[i](X_split[j], (h_t, c_t))
                X_split[j] = h_t
                if i == self.num_layers - 1:
                    out_features_split[j] = h_t
            out_h_split[i] = h_t
            out_c_split[i] = c_t
        out_features = ops.stack(tuple(out_features_split), 0)
        out_h = ops.stack(tuple(out_h_split), 0)
        out_c = ops.stack(tuple(out_c_split), 0)
        return out_features, (out_h, out_c)
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(init.randn(*(num_embeddings, embedding_dim), mean=0, std=1, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        x_one_hot = init.one_hot(self.num_embeddings, x, device=x.device, dtype=x.dtype)
        seq_len, bs = x.shape
        ret = x_one_hot.reshape((seq_len*bs, self.num_embeddings)) @ self.weight
        ret = ret.reshape((seq_len, bs, self.embedding_dim))
        return ret
        ### END YOUR SOLUTION
