from __future__ import annotations
import numpy as np

import ops
from graph import CalNode
from autograd import backward


class Tensor:
    data: np.ndarray
    requires_grad: bool
    graph: CalNode | None
    grad: np.ndarray | None

    def __init__(self, original: list | np.ndarray | Tensor, requires_grad=False):
        if isinstance(original, Tensor):
            self.data = original.data.copy()
        elif isinstance(original, np.ndarray):
            self.data = original.copy()
        elif isinstance(original, list):
            self.data = np.array(original)
        else:
            raise ValueError(f"Invalid input type {type(original)}")
        self.requires_grad = requires_grad
        self.graph = None
        self.grad = np.zeros_like(original) if requires_grad else None

    def zero_grad_(self):
        self.grad = np.zeros_like(self.grad)

    def requires_grad_(self, requires_grad=True):
        self.requires_grad = requires_grad
        return self

    def backward(self, grad: np.ndarray = np.ones([1])):
        backward(self, grad)

    @property
    def shape(self):
        return self.data.shape

    def __add__(self, other):
        return ops.add(self, other)

    def __mul__(self, other):
        return ops.mul(self, other)

    def __truediv__(self, other):
        return ops.true_divide(self, other)

    def __floordiv__(self, other):
        return ops.floor_divide(self, other)

    def __sub__(self, other):
        return ops.add(self, -other)

    def __matmul__(self, other):
        return ops.dot(self, other)

    def dot(self, other):
        return ops.dot(self, other)


def zeros(*shape):
    return Tensor(np.zeros(shape))


def zeros_like(tensor: Tensor):
    return zeros(*tensor.shape)


def ones(*shape):
    return Tensor(np.ones(shape))


def ones_like(tensor: Tensor):
    return ones(*tensor.shape)