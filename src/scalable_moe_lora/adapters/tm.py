"""TM-LoRA: shared A/B + learned expert-vector table added at the bottleneck.

Forward:
  hidden = A(x) + E_token,  where E_token = softmax-over-top-k sum of E rows
  out    = (alpha/r) * B(GELU(hidden))

Per-layer params: 2*d*r + K*r. Active rank per token = r (expert vectors bias
the r-dim bottleneck, don't expand it).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TMLoRA(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=32, dropout=0.0,
                 num_experts=8, top_k=4, **kwargs):
        super().__init__()
        self.scaling = alpha / rank
        self.top_k = min(top_k, num_experts)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.expert_vectors = nn.Parameter(torch.randn(num_experts, rank) * 0.01)
        self.router = nn.Linear(in_features, num_experts, bias=False)

        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        x_dropped = self.dropout(x)

        scores = self.router(x)                                         # (B, S, K)
        topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)                    # (B, S, k)

        selected = self.expert_vectors[topk_idx]                       # (B, S, k, r)
        E_token = (selected * topk_weights.unsqueeze(-1)).sum(dim=-2)  # (B, S, r)

        hidden = self.A(x_dropped) + E_token
        return self.B(F.gelu(hidden)) * self.scaling
