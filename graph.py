from typing import Any

import tensor
from enums import Ops


class CalNode:
    op: Ops
    output: tensor.Tensor
    inputs: list[tensor.Tensor]
    other: Any

    def __init__(self, op: Ops, inputs: list[tensor.Tensor], output: tensor.Tensor):
        self.op = op
        self.inputs = inputs
        self.output = output
