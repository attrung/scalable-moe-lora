"""MoE-LoRA (Luo et al. 2024) — stack-and-gather mixture of K LoRA experts.

K independent (A_i, B_i) expert pairs, each rank r, with a linear router selecting
top-k. All K expert outputs are computed, stacked, then top-k selected and softmax-
weighted. Activation memory scales linearly in K (shape (B, S, K, d)) — OOMs on
A100 80GB at K=64, which is why the granularity sweep uses RoutedLoRA (shared-
bottleneck) as its primary MoE-LoRA implementation. MoELoRA / RoutedLoRA /
DispatchMoELoRA are proven numerically equivalent at matched (K, r, top_k) —
see scripts/correctness_test.py.

Reference: Luo et al. 2024, arXiv:2402.12851.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELoRA(nn.Module):
    """K independent LoRA experts + linear router + stack-and-gather top-k."""

    def __init__(self, in_features, out_features, rank=4, alpha=32, dropout=0.0,
                 num_experts=8, top_k=2, **kwargs):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.experts_A = nn.ModuleList([
            nn.Linear(in_features, rank, bias=False) for _ in range(num_experts)
        ])
        self.experts_B = nn.ModuleList([
            nn.Linear(rank, out_features, bias=False) for _ in range(num_experts)
        ])
        self.router = nn.Linear(in_features, num_experts, bias=False)

        for A in self.experts_A:
            nn.init.kaiming_uniform_(A.weight, a=math.sqrt(5))
        for B in self.experts_B:
            nn.init.zeros_(B.weight)

    def forward(self, x):
        x_dropped = self.dropout(x)

        scores = self.router(x)                                          # (B, S, K)
        topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        self._last_routing_indices = topk_idx.detach()
        topk_weights = F.softmax(topk_vals, dim=-1)                     # (B, S, k)

        expert_outputs = torch.stack(
            [B(A(x_dropped)) for A, B in zip(self.experts_A, self.experts_B)],
            dim=-2,
        )                                                                # (B, S, K, d)

        idx = topk_idx.unsqueeze(-1).expand(-1, -1, -1, expert_outputs.shape[-1])
        selected = torch.gather(expert_outputs, dim=-2, index=idx)       # (B, S, k, d)
        output = (selected * topk_weights.unsqueeze(-1)).sum(dim=-2)     # (B, S, d)
        return output * self.scaling
