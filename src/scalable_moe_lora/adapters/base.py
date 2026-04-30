"""Standard LoRA adapter and LinearWithLoRA injection wrapper.

LoRA (Hu et al. 2021): freeze the pretrained weight W_0 and learn a low-rank
residual delta = B @ A in parallel. Per layer it adds 2*d*r trainable params,
no extra inference compute beyond the residual matmul.

LinearWithLoRA wraps any nn.Linear so that its forward becomes
  base(x) + adapter(x)
The adapter's forward runs in fp32 for numerical stability when the base
weights are in fp16; the result is cast back to the base dtype before the add.
"""

import math

import torch
import torch.nn as nn


class LoRA(nn.Module):
    """Standard low-rank adapter: out = (alpha/r) * B @ A @ x."""

    def __init__(self, in_features, out_features, rank=4, alpha=32, dropout=0.0, **kwargs):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.B(self.A(self.dropout(x))) * self.scaling


class LinearWithLoRA(nn.Module):
    """Adapter injection point: original(x) + lora(x)."""

    def __init__(self, original_linear, lora_module):
        super().__init__()
        self.original_linear = original_linear
        self.lora = lora_module

    def forward(self, x):
        base_out = self.original_linear(x)
        lora_out = self.lora(x.float()).to(base_out.dtype)
        return base_out + lora_out
