"""RoutedLoRA — shared-bottleneck routed LoRA with top-k gating.

Single shared A: (Kr, d) and B: (d, Kr). The Kr bottleneck dimensions are partitioned
into K expert groups of r columns each. Forward:
  z = A x  (one full matmul, always computes all Kr values)
  apply top-k gate over K blocks
  out = B (z masked)
Activation memory is (B, S, Kr) — constant in how K*r is split.

Scaling: alpha / top_k (this is what the granularity & router sweep used historically).

Routing delegated to `src/adapters/routers.py`. Six router types supported.

Load-balance auxiliary loss: the module stashes the Switch-Transformer LB term
on self._last_aux_loss per forward pass. The trainer collects aux loss across
all routed modules (RoutedLoRA + DispatchMoELoRA) via `collect_aux_loss`. The
LB loss prevents routing collapse at low top_k (e.g., K=8, k=2 with a linear
router diverged without it in an earlier run).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .routers import build_router, EarlySharedRouter


class RoutedLoRA(nn.Module):
    def __init__(self, in_features, out_features, rank=1, alpha=32, dropout=0.0,
                 num_experts=64, top_k=16, router_type="lowrank", router_dim=16,
                 **kwargs):
        super().__init__()
        self.num_experts = num_experts
        self.rank = rank
        self.top_k = top_k
        self.router_type = router_type
        self.scaling = alpha / top_k
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        bottleneck = num_experts * rank
        self.A = nn.Linear(in_features, bottleneck, bias=False)
        self.B = nn.Linear(bottleneck, out_features, bias=False)

        self.router = build_router(
            router_type, d=in_features, num_experts=num_experts,
            top_k=top_k, router_dim=router_dim, **kwargs,
        )

        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

        self._last_routing_indices = None
        self._last_topk_weights = None
        self._last_full_scores = None
        self._last_aux_loss = None
        self._teacher_full_scores = None
        self._last_distill_loss = None

    def forward(self, x):
        x_dropped = self.dropout(x)

        z = self.A(x_dropped)
        z = z.view(*z.shape[:-1], self.num_experts, self.rank)  # (B, S, K, r)

        topk_idx, topk_weights, full_scores = self.router(x_dropped)
        self._last_routing_indices = topk_idx.detach()
        self._last_topk_weights = topk_weights.detach()
        self._last_full_scores = full_scores.detach() if full_scores is not None else None

        # Distillation: KL(student || teacher) on the K-wide softmax distribution.
        # Teacher full_scores are deposited externally on _teacher_full_scores per
        # step (see src/train.py). Trainer sums _last_distill_loss across modules
        # via collect_distill_loss().
        if self._teacher_full_scores is not None and full_scores is not None:
            student_logp = F.log_softmax(full_scores.reshape(-1, K := self.num_experts), dim=-1)
            teacher_p = F.softmax(self._teacher_full_scores.reshape(-1, K), dim=-1)
            self._last_distill_loss = F.kl_div(
                student_logp, teacher_p, reduction="batchmean",
            )
        else:
            self._last_distill_loss = None

        # Build (..., K) gate; zero outside top-k.
        gate = torch.zeros(
            *x_dropped.shape[:-1], self.num_experts,
            device=x_dropped.device, dtype=topk_weights.dtype,
        )
        gate.scatter_(-1, topk_idx, topk_weights)

        # ---- Load-balance aux loss (Switch Transformer) ----
        # f_i = fraction of tokens with expert i in their top-k.
        # p_i = mean router softmax probability for expert i (over all K).
        # Flatten token dims so f, p are (K,).
        K = self.num_experts
        flat_topk = topk_idx.reshape(-1, self.top_k)
        onehot = torch.zeros(
            flat_topk.shape[0], K, device=x.device, dtype=topk_weights.dtype,
        )
        onehot.scatter_(1, flat_topk, 1.0)
        f = onehot.mean(0)
        if full_scores is not None:
            p = F.softmax(full_scores.reshape(-1, K), dim=-1).mean(0)
        else:
            # hierarchical router doesn't expose full K-wide scores; fall back
            # to top-k softmax weights scattered to K (approximate).
            p_onehot = torch.zeros_like(onehot)
            p_onehot.scatter_(1, flat_topk, topk_weights.reshape(-1, self.top_k))
            p = p_onehot.mean(0)
        self._last_aux_loss = K * (f * p).sum()

        z_gated = z * gate.unsqueeze(-1)                       # (B, S, K, r)
        z_flat = z_gated.flatten(start_dim=-2)                  # (B, S, Kr)
        return self.B(z_flat) * self.scaling
