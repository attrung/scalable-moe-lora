"""DispatchMoELoRA — "normal MoE on LoRA" with dispatch forward pass.

Structure matches production MoE (Mixtral, DeepSeek-V3, Switch Transformer):
  - K independent expert pairs (A_i, B_i), each rank r.
  - Router scores x -> K experts, top-k selected with softmax gate.
  - Only the selected experts compute on their routed tokens (dispatch), via
    a vectorized padded-bmm: sort tokens by expert, pack into (K, max_N, d),
    run two bmm kernels, scatter back. 2 kernel launches per LoRA injection
    regardless of K.

Contrast with RoutedLoRA: RoutedLoRA shares A,B and masks at the bottleneck;
DispatchMoELoRA has K separate (A_i, B_i) and only routes selected tokens.

Scaling: alpha / rank (standard MoE-LoRA convention; differs from RoutedLoRA's
alpha / top_k — documented explicitly so cross-architecture comparisons account
for it).

Load-balance auxiliary loss (Switch Transformer):
  L_lb = K * sum_i f_i * p_i
where f_i is the fraction of tokens whose top-k contains expert i, p_i is the
mean router softmax probability for expert i. Stashed on self._last_aux_loss;
the trainer collects it across all DispatchMoELoRA modules via `collect_aux_loss`.

Routers: all 6 from `routers.py` (linear / lowrank / cosine / hierarchical /
product_key / early_shared) supported via `router_type`.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .routers import build_router


class DispatchMoELoRA(nn.Module):
    def __init__(self, in_features, out_features, rank=1, alpha=32, dropout=0.0,
                 num_experts=64, top_k=16, router_type="linear", router_dim=16,
                 **kwargs):
        super().__init__()
        assert out_features == in_features, (
            "DispatchMoELoRA assumes square base projection (q_proj/v_proj in LLaMA)."
        )
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.top_k = top_k
        self.rank = rank
        self.router_type = router_type
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Stacked expert params: (K, r, d_in) and (K, d_out, r).
        self.A = nn.Parameter(torch.empty(num_experts, rank, in_features))
        self.B = nn.Parameter(torch.empty(num_experts, out_features, rank))

        self.router = build_router(
            router_type, d=in_features, num_experts=num_experts,
            top_k=top_k, router_dim=router_dim, **kwargs,
        )

        for k_idx in range(num_experts):
            nn.init.kaiming_uniform_(self.A[k_idx], a=math.sqrt(5))
        nn.init.zeros_(self.B)

        self._last_routing_indices = None
        self._last_aux_loss = None
        self._last_load_balance_stats = None

    def forward(self, x):
        shape_prefix = x.shape[:-1]
        N = 1
        for s in shape_prefix:
            N *= s
        d = self.in_features
        K = self.num_experts
        k = self.top_k

        x_dropped = self.dropout(x)
        x_flat = x_dropped.reshape(N, d)

        # ---- Router ----
        topk_idx, gates, full_scores = self.router(x_flat)
        # topk_idx: (N, k) long; gates: (N, k) float; full_scores: (N, K) or None.
        self._last_routing_indices = topk_idx.detach().reshape(*shape_prefix, k)

        # ---- Dispatch pack ----
        token_idx = torch.arange(N, device=x.device).unsqueeze(-1).expand(N, k)
        token_idx = token_idx.reshape(-1)                    # (N*k,)
        flat_expert = topk_idx.reshape(-1)                   # (N*k,)
        flat_gate = gates.reshape(-1)                        # (N*k,)

        sort_perm = flat_expert.argsort()
        expert_s = flat_expert[sort_perm]
        token_s = token_idx[sort_perm]
        gate_s = flat_gate[sort_perm]

        counts = torch.bincount(expert_s, minlength=K)
        max_N = int(counts.max().item())
        start_offsets = torch.cat([
            torch.zeros(1, device=x.device, dtype=counts.dtype),
            counts.cumsum(0)[:-1],
        ])
        row_in_block = (
            torch.arange(expert_s.shape[0], device=x.device) - start_offsets[expert_s]
        )

        packed = x_flat.new_zeros(K, max_N, d)
        packed[expert_s, row_in_block] = x_flat[token_s]

        # ---- Two bmms ----
        bottleneck = torch.bmm(packed, self.A.transpose(1, 2))          # (K, max_N, r)
        expert_out = torch.bmm(bottleneck, self.B.transpose(1, 2))      # (K, max_N, d)

        # ---- Gate + scatter-add back ----
        gated = expert_out[expert_s, row_in_block] * gate_s.unsqueeze(-1)
        out_flat = x_flat.new_zeros(N, self.out_features)
        out_flat.index_add_(0, token_s, gated)

        # ---- Load-balance aux loss (Switch Transformer) ----
        # For routers that don't expose a full K-wide score tensor (e.g. hierarchical),
        # fall back to a one-hot-of-topk approximation for p_i — still a valid LB signal.
        onehot = torch.zeros(N, K, device=x.device, dtype=gates.dtype)
        onehot.scatter_(1, topk_idx, 1.0)
        f = onehot.mean(0)
        if full_scores is not None:
            p = F.softmax(full_scores, dim=-1).mean(0)
        else:
            # Fallback: use softmaxed top-k weights scattered to K (approximate).
            p_onehot = torch.zeros(N, K, device=x.device, dtype=gates.dtype)
            p_onehot.scatter_(1, topk_idx, gates)
            p = p_onehot.mean(0)
        self._last_aux_loss = K * (f * p).sum()
        self._last_load_balance_stats = {
            "f_min": f.min().detach(), "f_max": f.max().detach(),
            "f_std": f.std().detach(), "count_max": counts.max().detach(),
            "count_mean": counts.float().mean().detach(),
        }

        return out_flat.reshape(*shape_prefix, self.out_features) * self.scaling


def collect_aux_loss(model):
    """Sum _last_aux_loss across routed adapters (RoutedLoRA + DispatchMoELoRA).

    Any module with a non-None `_last_aux_loss` attribute contributes. Returns
    a zero scalar on the model's parameter device if no routed modules exist
    (so the trainer can always add it unconditionally).
    """
    total = None
    for m in model.modules():
        aux = getattr(m, "_last_aux_loss", None)
        if aux is not None:
            total = aux if total is None else total + aux
    if total is None:
        device = next(model.parameters()).device
        return torch.zeros((), device=device)
    return total


def collect_distill_loss(model):
    """Sum _last_distill_loss across RoutedLoRA modules (set when teacher scores
    are deposited via set_teacher_scores). Returns zero scalar if no module had
    a teacher set this step.
    """
    total = None
    for m in model.modules():
        d = getattr(m, "_last_distill_loss", None)
        if d is not None:
            total = d if total is None else total + d
    if total is None:
        device = next(model.parameters()).device
        return torch.zeros((), device=device)
    return total


def set_teacher_scores(student_model, teacher_scores_by_name):
    """Deposit teacher full_scores on each student RoutedLoRA module by matching
    qualified module names. teacher_scores_by_name: dict[name -> Tensor or None].
    Modules without a matching name get None (skip distillation this step).
    """
    for name, m in student_model.named_modules():
        if hasattr(m, "_teacher_full_scores"):
            m._teacher_full_scores = teacher_scores_by_name.get(name)


def collect_full_scores(model):
    """Snapshot {module_name: full_scores tensor} across all RoutedLoRA modules.
    Detached but on the same device as the source module."""
    out = {}
    for name, m in model.named_modules():
        scores = getattr(m, "_last_full_scores", None)
        if scores is not None:
            out[name] = scores
    return out
