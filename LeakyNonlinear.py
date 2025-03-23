#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2022/11/27 6:56
@File:          LeakyPFLU.py
'''

import random
from typing import Any, Tuple
from torch import Tensor
import torch
from torch import nn
from torch.autograd import Function



class PFLUFunction(Function):
    '''引自论文 PFLU and FPFLU：Two novel non-monotonic activation functions in convolutional neural networks
    '''
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        sqrt_s_y = x * torch.rsqrt(torch.square(x) + 1)
        y = x * (1 + sqrt_s_y) * 0.5
        return y

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Tensor], output: Tensor) -> None:
        x, = inputs
        sqrt_s_y = x * torch.rsqrt(torch.square(x) + 1)
        ctx.save_for_backward(sqrt_s_y)

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Tensor | None:
        sqrt_s_y, = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = (1 + (2 - torch.square(sqrt_s_y)) * sqrt_s_y) * 0.5
            grad_x = grad_output * grad_x
        return grad_x


class LeakyPFLU(nn.Module):
    def __init__(self, lamb: float = 0.01,
                 momentum: float = 0.00125,
                 randomized: bool = True,
                 inplace: bool = True,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.lamb = lamb
        self.momentum = momentum
        self.randomized = randomized
        self.inplace = inplace

        if randomized:
            self.register_buffer("running_lamb", torch.tensor(lamb, **factory_kwargs))
        else:
            self.register_buffer("running_lamb", None)

    def forward(self, inputs: Tensor) -> Tensor:
        if not self.randomized:
            lamb = self.lamb
            return (1 - lamb) * PFLUFunction.apply(inputs) + lamb * inputs

        if self.training:
            lamb = random.uniform(0., self.lamb * 2)
            self.running_lamb.mul_(1 - self.momentum).add_(torch.as_tensor(lamb), alpha=self.momentum)
        else:
            lamb = self.running_lamb
        y = (1 - lamb) * PFLUFunction.apply(inputs) + lamb * inputs
        return y

    def extra_repr(self) -> str:
        return f"lamb={self.lamb}, momentum={self.momentum}, randomized={self.randomized}, inplace={self.inplace}"