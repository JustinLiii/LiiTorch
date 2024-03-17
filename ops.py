import numpy as np

from tensor import Tensor
from enum import Enum
from graph import CalNode


class Ops(Enum):
    # 注意加入枚举
    add: 0
    mul: 1
    dot: 2
    power: 3
    log: 4
    true_divide: 5
    floor_divide: 6
    maximum: 7
    concat: 8


# TODO: Broadcast
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
    ret.graph = CalNode(Ops.mul, [tensor1, tensor2], ret)
    return ret


def floor_divide(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    try:
        result = tensor1.data // tensor2.data
    except ValueError as e:
        raise ValueError(e) from e

    ret = Tensor(result)
    ret.graph = CalNode(Ops.mul, [tensor1, tensor2], ret)
    return ret


def dot(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    try:
        result = np.dot(tensor1.data, tensor2.data)
    except ValueError as e:
        raise ValueError(e) from e

    ret = Tensor(result)
    ret.graph = CalNode(Ops.dot, [tensor1, tensor2], ret)
    return ret


def power(tensor: Tensor, base: Tensor | float = np.e) -> Tensor:
    try:
        result = np.power(tensor.data, base)
    except ValueError as e:
        raise ValueError(e) from e

    ret = Tensor(result)
    ret.graph = CalNode(Ops.power, [tensor, base], ret)
    return ret


def log(tensor: Tensor) -> Tensor:
    try:
        result = np.log(tensor.data)
    except ValueError as e:
        raise ValueError(e) from e

    ret = Tensor(result)
    ret.graph = CalNode(Ops.log, [tensor], ret)
    return ret


def maximum(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    try:
        result = np.max(tensor1.data, tensor2.data)
    except ValueError as e:
        raise ValueError(e) from e

    ret = Tensor(result)
    ret.graph = CalNode(Ops.max, [tensor1, tensor2], ret)
    return ret


def concat(tensors: list[Tensor], axis: int):
    try:
        result = np.concatenate([tensor.data for tensor in tensors], axis=axis)
    except ValueError as e:
        raise ValueError(e) from e

    ret = Tensor(result)
    ret.graph = CalNode(Ops.concat, tensors, ret)
    ret.graph.other = axis
    return ret
