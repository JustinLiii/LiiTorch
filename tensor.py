from __future__ import annotations

import numpy as np
from graph import CalNode


class Tensor:
    data: np.ndarray
    requires_grad: bool
    graph: CalNode | None
    grad: np.ndarray | None

    def __init__(self, original, requires_grad=False):
        if isinstance(original, Tensor):
            self.data = original.data.copy()
        elif isinstance(original, np.ndarray):
            self.data = original.copy()
        elif isinstance(original, list):
            self.data = np.array(original)
        elif isinstance(original, (int, float, complex, np.number)):
            self.data = np.array([original])
        else:
            raise ValueError(f"Invalid input type {type(original)}")
        self.requires_grad = requires_grad
        self.graph = None
        self.grad = np.zeros_like(original) if requires_grad else None

    def zero_grad_(self):
        self.grad = np.zeros_like(self.grad)

    def requires_grad_(self, requires_grad=True):
        self.requires_grad = requires_grad

    def backward(self, grad: np.ndarray = np.ones([1])):
        from autograd import backward
        backward(self, grad)

    @property
    def shape(self):
        return self.data.shape

    def __add__(self, other):
        from ops import add
        return add(self, other)

    def __mul__(self, other):
        from ops import mul
        return mul(self, other)

    def __truediv__(self, other):
        from ops import true_divide
        return true_divide(self, other)

    def __floordiv__(self, other):
        from ops import floor_divide
        return floor_divide(self, other)

    def __sub__(self, other):
        from ops import add
        return add(self, -other)

    def __matmul__(self, other):
        from ops import dot
        return dot(self, other)

    def dot(self, other):
        from ops import dot
        return dot(self, other)

    def __str__(self):
        return (f"Tensor(shape={self.shape})\n"
                + f"requires_grad={self.requires_grad}\n"
                + f"data={self.data}" + "\n"
                + f"grad={self.grad}\n")


def zeros(shape: tuple[int, ...]):
    return Tensor(np.zeros(shape))


def zeros_like(tensor: Tensor):
    return zeros(tensor.shape)


def ones(shape: tuple[int, ...]):
    return Tensor(np.ones(shape))


def ones_like(tensor: Tensor):
    return ones(tensor.shape)
