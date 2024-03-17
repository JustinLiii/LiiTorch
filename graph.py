from typing import Any

from tensor import Tensor
from ops import Ops


class CalNode:
    op: Ops
    output: Tensor
    inputs: list[Tensor]
    other : Any

    def __init__(self, op: Ops, inputs: list[Tensor], output: Tensor):
        self.op = op
        self.inputs = inputs
        self.output = output
