"""MoE-LoRA adapter.

K LoRA experts of rank `r`, with a router that picks `top_k` experts per token
and softmax-normalizes their scores into gate weights. Implemented as a single
shared `A: (Kr, d)` and `B: (d, Kr)`, with the `Kr` bottleneck partitioned into
K expert groups of `r` columns each and the top-k gate applied at the
bottleneck:

  z = A x                              # one matmul, all Kr values
  apply top-k gate over the K blocks   # zero non-selected blocks, scale chosen
  out = (alpha / top_k) · B z          # one matmul; activation memory O(B·S·K·r)

Routing is delegated to `routers.py`; nine router types are supported
(`build_router`).

Auxiliary losses stashed per forward and collected by the trainer:
  * `_last_aux_loss`     — Switch-Transformer load balance: K·sum(f·p)
                            where f_i is the fraction of tokens whose top-k
                            includes expert i and p_i is the mean router
                            softmax probability for expert i. Mandatory at
                            low `top_k` to prevent routing collapse.
  * `_last_distill_loss` — KL(student_softmax(full_scores) || teacher_softmax)
                            (only set when `_teacher_full_scores` is deposited
                            externally; see `train.py`).
  * `_last_routing_indices`, `_last_topk_weights`, `_last_full_scores`
                          — captured for post-hoc routing analysis (Part D).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .routers import build_router


class MoELoRA(nn.Module):
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

        K = self.num_experts

        # Distillation: KL(student || teacher) on the K-wide softmax distribution.
        # Teacher full_scores are deposited externally per step (see train.py).
        if self._teacher_full_scores is not None and full_scores is not None:
            student_logp = F.log_softmax(full_scores.reshape(-1, K), dim=-1)
            teacher_p   = F.softmax(self._teacher_full_scores.reshape(-1, K), dim=-1)
            self._last_distill_loss = F.kl_div(student_logp, teacher_p, reduction="batchmean")
        else:
            self._last_distill_loss = None

        # Build (..., K) gate; zero outside top-k.
        gate = torch.zeros(
            *x_dropped.shape[:-1], K,
            device=x_dropped.device, dtype=topk_weights.dtype,
        )
        gate.scatter_(-1, topk_idx, topk_weights)

        # Load-balance aux loss (Switch Transformer): K · sum_i f_i · p_i.
        # EarlySharedRouter followers reuse the owner's routing decision verbatim,
        # so their (f, p) is identical to the owner's. Counting it again would
        # multiply the effective aux_loss_coef by the number of injection sites
        # (32× at qv on a 16-layer model). Followers set _last_aux_loss=None;
        # collect_aux_loss skips them.
        if getattr(self.router, "is_owner", True):
            flat_topk = topk_idx.reshape(-1, self.top_k)
            onehot = torch.zeros(flat_topk.shape[0], K, device=x.device, dtype=topk_weights.dtype)
            onehot.scatter_(1, flat_topk, 1.0)
            f = onehot.mean(0)
            if full_scores is not None:
                p = F.softmax(full_scores.reshape(-1, K), dim=-1).mean(0)
            else:
                # hierarchical router doesn't expose full K-wide scores; fall back to
                # top-k softmax weights scattered to K (approximate but valid signal).
                p_onehot = torch.zeros_like(onehot)
                p_onehot.scatter_(1, flat_topk, topk_weights.reshape(-1, self.top_k))
                p = p_onehot.mean(0)
            self._last_aux_loss = K * (f * p).sum()
        else:
            self._last_aux_loss = None

        z_gated = z * gate.unsqueeze(-1)                       # (B, S, K, r)
        z_flat = z_gated.flatten(start_dim=-2)                  # (B, S, Kr)
        return self.B(z_flat) * self.scaling


def collect_aux_loss(model):
    """Sum `_last_aux_loss` across all MoELoRA modules. Returns a zero scalar on the
    model's parameter device if no module has emitted one (so the trainer can add
    it unconditionally)."""
    total = None
    for m in model.modules():
        aux = getattr(m, "_last_aux_loss", None)
        if aux is not None:
            total = aux if total is None else total + aux
    if total is None:
        return torch.zeros((), device=next(model.parameters()).device)
    return total


def collect_distill_loss(model):
    """Sum `_last_distill_loss` across modules. Returns zero scalar when no teacher
    is set this step."""
    total = None
    for m in model.modules():
        d = getattr(m, "_last_distill_loss", None)
        if d is not None:
            total = d if total is None else total + d
    if total is None:
        return torch.zeros((), device=next(model.parameters()).device)
    return total


def collect_full_scores(model):
    """Snapshot {qualified_module_name: full_scores tensor} for every MoELoRA
    module. Detached but on the same device as the source module — intended to
    be deposited onto a student model via `set_teacher_scores`."""
    return {
        name: scores
        for name, m in model.named_modules()
        if (scores := getattr(m, "_last_full_scores", None)) is not None
    }


def set_teacher_scores(student_model, teacher_scores_by_name):
    """Deposit teacher `full_scores` on each student MoELoRA module by qualified
    name. Modules without a matching teacher entry get `None` (skip distillation
    this step)."""
    for name, m in student_model.named_modules():
        if hasattr(m, "_teacher_full_scores"):
            m._teacher_full_scores = teacher_scores_by_name.get(name)
