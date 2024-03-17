import numpy as np

from tensor import Tensor
from graph import CalNode
from enums import Ops


def add(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    try:
        result = tensor1.data + tensor2.data
    except ValueError as e:
        raise ValueError(e) from e

    ret = Tensor(result)
    ret.graph = CalNode(Ops.add, [tensor1, tensor2], ret)
    return ret


def mul(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    try:
        result = tensor1.data * tensor2.data
    except ValueError as e:
        raise ValueError(e) from e

    ret = Tensor(result)
    ret.graph = CalNode(Ops.mul, [tensor1, tensor2], ret)
    return ret


def true_divide(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    try:
        result = tensor1.data / tensor2.data
    except ValueError as e:
        raise ValueError(e) from e

    ret = Tensor(result)
    ret.graph = CalNode(Ops.true_divide, [tensor1, tensor2], ret)
    return ret


def floor_divide(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    try:
        result = tensor1.data // tensor2.data
    except ValueError as e:
        raise ValueError(e) from e

    ret = Tensor(result)
    ret.graph = CalNode(Ops.floor_divide, [tensor1, tensor2], ret)
    return ret


def dot(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    try:
        result = np.dot(tensor1.data, tensor2.data)
    except ValueError as e:
        raise ValueError(e) from e

    ret = Tensor(result)
    ret.graph = CalNode(Ops.dot, [tensor1, tensor2], ret)
    return ret


def power(tensor1: Tensor, base: Tensor | float = np.e) -> Tensor:
    try:
        result = np.power(tensor1.data, base)
    except ValueError as e:
        raise ValueError(e) from e

    ret = Tensor(result)
    ret.graph = CalNode(Ops.power, [tensor1, base], ret)
    return ret


def log(tensor1: Tensor) -> Tensor:
    try:
        result = np.log(tensor1.data)
    except ValueError as e:
        raise ValueError(e) from e

    ret = Tensor(result)
    ret.graph = CalNode(Ops.log, [tensor1], ret)
    return ret


def maximum(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    try:
        result = np.max(tensor1.data, tensor2.data)
    except ValueError as e:
        raise ValueError(e) from e

    ret = Tensor(result)
    ret.graph = CalNode(Ops.maximum, [tensor1, tensor2], ret)
    return ret


def concat(tensors: list[Tensor], axis: int):
    try:
        result = np.concatenate([t.data for t in tensors], axis=axis)
    except ValueError as e:
        raise ValueError(e) from e

    ret = Tensor(result)
    ret.graph = CalNode(Ops.concat, tensors, ret)
    ret.graph.other = axis
    return ret
