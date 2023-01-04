"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad, )


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return (self.scalar*array_api.power(a, self.scalar-1)*out_grad, )
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return (out_grad / b, (-1)*a*out_grad/(b**2))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad/self.scalar,)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        ndim = len(a.shape)
        args_axes = list(range(ndim))
        if self.axes is None:
            args_axes[-1] = ndim - 2
            args_axes[-2] = ndim - 1
        else:
            args_axes[self.axes[0]] = self.axes[1]
            args_axes[self.axes[1]] = self.axes[0]
        return array_api.transpose(a, axes=tuple(args_axes))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return (transpose(out_grad, axes=self.axes),)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)

class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # print('reshape forward: a.shape: ', a.shape)
        # print('reshape forward: self.shape: ', self.shape)
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        if a.shape == self.shape:
            return (out_grad, )
        # print('    {reshape backward} out_grad shape:', out_grad.shape)
        # print('    {reshape backward} self.shape:', self.shape)
        # print('    {reshape backward} a shape:', a.shape)
        return (reshape(out_grad, a.shape),)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 先按广播的维度求和，再reshape到原始shape
        # 一维张量默认是行向量，一维张量(x, )可以广播到(y, x)，不能广播到(x, y)
        a = node.inputs[0]
        if a.shape == self.shape:
            return (out_grad, )
        new_a_shape = [1] * (len(self.shape)-len(a.shape)) + list(a.shape)
        axes = tuple([i for i in range(len(self.shape)) if (i>=len(new_a_shape) or new_a_shape[i]!=self.shape[i])])
        grad_a = summation(out_grad, axes)
        grad_a = reshape(grad_a, a.shape)
        return (grad_a, )
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if isinstance(self.axes, tuple):
            for i, axis in enumerate(sorted(list(self.axes))):
                a = a.sum(axis-i)
            return a
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 直接广播到原始shape
        a = node.inputs[0]
        if isinstance(self.axes, int):
            self.axes = (self.axes, )
        if self.axes is None:
            reshape_size = [1 for i in range(len(a.shape))]
            grad_a = reshape(out_grad, reshape_size)
            grad_a = broadcast_to(grad_a, a.shape)  
        else:
            reshape_size = [1 if (i in self.axes) else a.shape[i] for i in range(len(a.shape))]
            grad_a = reshape(out_grad, reshape_size)
            grad_a = broadcast_to(grad_a, a.shape)  
        return (grad_a,)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        grad_a, grad_b = (matmul(out_grad, transpose(b)), matmul(transpose(a), out_grad))
        if a.shape != grad_a.shape:
            grad_a = summation(grad_a, axes=tuple(range(len(grad_a.shape)-len(a.shape))))
        if b.shape != grad_b.shape:
            grad_b = summation(grad_b, axes=tuple(range(len(grad_b.shape)-len(b.shape))))
        return (grad_a, grad_b)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (-1*out_grad, )
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return (out_grad / a, )
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return (exp(a)*out_grad, )
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        data = node.inputs[0].realize_cached_data()
        return (out_grad * Tensor(data>0, device=node.inputs[0].device), )
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.amax(Z, axis=self.axes, keepdims=True)
        res = array_api.log(array_api.sum(array_api.exp(Z-max_Z.broadcast_to(Z.shape)), axis=self.axes, keepdims=True)) + max_Z
        if self.axes is None:
            return res.reshape((1, ))
        if isinstance(self.axes, int):
            self.axes = [self.axes]
        out_shape = tuple([Z.shape[i] for i in range(len(Z.shape)) if i not in self.axes])
        return res.compact().reshape(out_shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Z = node.inputs[0]
        # if self.axes:
        #     shape = [1] * len(Z.shape)
        #     s = set(self.axes)
        #     j = 0
        #     for i in range(len(shape)):
        #         if i not in s:
        #             shape[i] = node.shape[j]
        #             j += 1
        #     node_new = node.reshape(shape)
        #     grad_new = out_grad.reshape(shape)
        # else:
        #     node_new = node
        #     grad_new = out_grad
        # return (grad_new * exp(Z - node_new), )
        Z = node.inputs[0]
        if self.axes is not None:
            shape = [1] * len(Z.shape)
            if isinstance(self.axes, int):
                s = set([self.axes])
            else:
                s = set(self.axes)
            j = 0
            for i in range(len(shape)):
                if i not in s:
                    shape[i] = node.shape[j]
                    j += 1
            node_new = node.reshape(shape)
            grad_new = out_grad.reshape(shape)
        else:
            node_new = node
            grad_new = out_grad
        return grad_new.broadcast_to(Z.shape) * exp(Z - node_new.broadcast_to(Z.shape))

        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return (out_grad * (1 - tanh(a)**2), )
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        new_shape = list(args[0].shape)
        # print('old shape: ', new_shape)
        new_shape.insert(self.axis, 1)
        # print('new shape: ', new_shape)
        tensor_list = [i.compact().reshape(new_shape).compact() for i in args]
        out_shape = [new_shape[i] if i != self.axis else new_shape[i]*len(args) for i in range(len(new_shape))]
        # print('out shape: ', out_shape)
        out = array_api.empty(tuple(out_shape), device=args[0].device)
        sl = []
        for i in range(len(out_shape)):
            if i == self.axis:
                sl.append(0)
            else:
                sl.append(slice(out_shape[i]))
        for i, ts in enumerate(tensor_list):
            sl[self.axis] = i
            out[tuple(sl)] = ts
        return out
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (split(out_grad, self.axis), )
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        out_shape = [A.shape[i] for i in range(len(A.shape)) if i!=self.axis]
        out_tensors = [array_api.empty(tuple(out_shape), device=A.device) for i in range(A.shape[self.axis])]
        sl = []
        for i in range(len(A.shape)):
            if i == self.axis:
                sl.append(0)
            else:
                sl.append(slice(A.shape[i]))
        for i in range(len(out_tensors)):
            sl[self.axis] = i
            out_tensors[i] = A[tuple(sl)].compact().reshape(out_shape).compact()
        return tuple(out_tensors)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if isinstance(out_grad, Tensor):
            return (stack((out_grad, ), self.axis), )
        return (stack(tuple(out_grad), self.axis), )
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (flip(out_grad, self.axes), )
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        out_shape = [a.shape[i] if i not in self.axes else a.shape[i]*(self.dilation+1) for i in range(len(a.shape))]
        out = array_api.full(tuple(out_shape), 0, device=a.device)
        sl = []
        for i in range(len(a.shape)):
            if i in self.axes:
                sl.append(slice(0, out_shape[i], self.dilation+1))
            else:
                sl.append(slice(out_shape[i]))
        out[tuple(sl)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (undilate(out_grad, self.axes, self.dilation), )
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        out_shape = [a.shape[i] if i not in self.axes else int(a.shape[i]/(self.dilation+1)) for i in range(len(a.shape))]
        out = array_api.empty(tuple(out_shape), device=a.device)
        sl = []
        for i in range(len(a.shape)):
            if i in self.axes:
                sl.append(slice(0, a.shape[i], self.dilation+1))
            else:
                sl.append(slice(out_shape[i]))
        out = a[tuple(sl)]
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (dilation(out_grad, self.axes, self.dilation), )
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


from numpy.lib.stride_tricks import as_strided
class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        A_pad = A.pad(((0,0), (self.padding, self.padding), (self.padding, self.padding), (0,0))).numpy()
        
        N, H, W, C_in = A_pad.shape
        K, _, _, C_out = B.shape
        # print('N={}, H={}, W={}, C_in={}, K={}, C_out={}'.format(N, H, W, C_in, K, C_out))
        Ns, Hs, Ws, Cs = A_pad.strides
        inner_dim = K * K * C_in
        # print('inner_dim: ', inner_dim)
        out_H = int((H-K+1)/self.stride)
        out_W = int((W-K+1)/self.stride)
        A_pad = as_strided(A_pad, shape=(N, out_H, out_W, K, K, C_in),
                    strides=(Ns, Hs*self.stride, Ws*self.stride, Hs, Ws, Cs)).reshape(-1, inner_dim)
        A_pad = NDArray(A_pad, device=A.device)
        # print('inner_dim: ', inner_dim)
        out = array_api.matmul(A_pad, (B.compact().reshape((inner_dim, C_out))))
        return array_api.reshape(out, (N, out_H, out_W, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs
        kernel_size = W.shape[0]
        # print('X shape: ', X.shape)
        # print('W shape: ', W.shape)
        # print('stride: ', self.stride)
        # print('padding: ', self.padding)
        # print('out shape: ', out_grad.shape)
        # calc X.grad
        W_flip = transpose(flip(W, axes=(0, 1)),(2, 3))
        # print('W_flip shape: ', W_flip.shape)
        new_padding = kernel_size - self.padding - 1
        # print('new_padding: ', new_padding)
        if self.stride > 1:
            out_grad_dilation = dilate(out_grad, axes=(1, 2), dilation=self.stride-1)
        else:
            out_grad_dilation = out_grad
        # print('out_grad_dilation shape: ', out_grad_dilation.shape)
        X_grad = conv(out_grad_dilation, W_flip, padding=new_padding)
        # calc W.grad
        # X: NHWCin -> CinHWN
        # W: NH'W'Cout -> H'W'NCout
        W_grad = conv(transpose(transpose(transpose(transpose(transpose(X,(3,2)),(2,1)),(1,0)),(1,2)),(2,3)), \
            transpose(transpose(out_grad_dilation, (0,1)), (1,2)), \
            padding=self.padding)
        # print('X_grad shape: ', X_grad.shape)
        # print('W_grad shape: ', W_grad.shape)
        return (X_grad, transpose(transpose(W_grad,(0,1)),(1,2)))
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



