import numpy as np

from tensor import Tensor
from ops import Ops


def backward(tensor: Tensor, grad: np.ndarray = np.ones([1])):
    # TODO: Broadcast
    assert grad.shape == tensor.shape
    if tensor.requires_grad:
        tensor.grad = grad if tensor.grad is None else tensor.grad + grad

    if tensor.graph is None:
        return

    if tensor.graph.op == Ops.add:
        for o in tensor.graph.inputs:
            backward(o, grad)

    elif tensor.graph.op == Ops.mul:
        for i, o in enumerate(tensor.graph.inputs):
            backward(o, grad * tensor.graph.inputs[1 - i].data)

    elif tensor.graph.op == Ops.dot:
        # TODO: 多维扩展
        backward(tensor.graph.inputs[0], grad @ tensor.graph.inputs[1].data.T)
        backward(tensor.graph.inputs[1], tensor.graph.inputs[0].data.T @ grad)

    elif tensor.graph.op == Ops.power:
        backward(tensor.graph.inputs[0],
                 grad * np.power(tensor.graph.inputs[0].data, tensor.graph.inputs[1].data - 1)
                 * tensor.graph.inputs[1].data)
        backward(tensor.graph.inputs[1],
                 grad * np.power(tensor.graph.inputs[0].data, tensor.graph.inputs[1].data)
                 * np.log(tensor.graph.inputs[0].data))

    elif tensor.graph.op == Ops.log:
        backward(tensor.graph.inputs[0], grad / tensor.graph.inputs[0].data)

    elif tensor.graph.op == Ops.true_divide:
        backward(tensor.graph.inputs[0], grad / tensor.graph.inputs[1].data)
        backward(tensor.graph.inputs[1], -grad * tensor.graph.inputs[0].data / np.power(tensor.graph.inputs[1].data, 2))

    elif tensor.graph.op == Ops.maximum:
        backward(tensor.graph.inputs[0], grad * int(tensor.graph.inputs[0].data >= tensor.graph.inputs[1].data))
        backward(tensor.graph.inputs[1], grad * int(tensor.graph.inputs[0].data < tensor.graph.inputs[1].data))

    elif tensor.graph.op == Ops.concat:
        axis = tensor.graph.other
        splits = [t.shape[axis] for t in tensor.graph.inputs]
        splits.pop()
        grads = np.split(grad, splits, axis)
        for o, g in zip(tensor.graph.inputs, grads):
            backward(o, g)

    else:
        raise ValueError(f"Op not supported {tensor.graph.op}")
