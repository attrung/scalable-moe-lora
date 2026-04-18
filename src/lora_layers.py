"""LoRA module variants.

Includes:
- Standard LoRA (Hu et al., 2021)
- Non-Linear LoRA (GELU activation between A and B)
- Frozen-Half LoRA (A frozen, only B trained)
- MoE-LoRA (Luo et al., 2024) — K independent expert pairs, linear routing
- TM-LoRA (Token-Modulated LoRA) — shared A/B with additive expert vectors
- LinearWithLoRA — wraps a frozen nn.Linear with an additive LoRA branch

The RoutedLoRA class (unified shared-bottleneck routed LoRA with six router
types) is added in later commits as the study progresses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Baseline + Ablations
# ---------------------------------------------------------------------------

class LoRA(nn.Module):
    """Standard LoRA: linear down-projection A, linear up-projection B."""

    def __init__(self, in_features, out_features, rank=4, alpha=32, dropout=0.0, **kwargs):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.B(self.A(self.dropout(x))) * self.scaling


class NonLinearLoRA(nn.Module):
    """GELU activation between A and B (ablation variant)."""

    def __init__(self, in_features, out_features, rank=4, alpha=32, dropout=0.0, **kwargs):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.B(F.gelu(self.A(self.dropout(x)))) * self.scaling


class FrozenHalfLoRA(nn.Module):
    """A is frozen (random projection); only B is trained (ablation variant)."""

    def __init__(self, in_features, out_features, rank=4, alpha=32, dropout=0.0, **kwargs):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)
        self.A.weight.requires_grad = False

    def forward(self, x):
        return self.B(self.A(self.dropout(x))) * self.scaling


# ---------------------------------------------------------------------------
# MoE-style LoRA variants
# ---------------------------------------------------------------------------

class MoELoRA(nn.Module):
    """MoE-LoRA: K independent LoRA experts with top-k linear routing.

    Reference: Luo et al. (2024). MoELoRA: Contrastive Learning Guided
    Mixture of Experts on PEFT for LLMs. arXiv:2402.12851.

    Each expert is an independent (A_i, B_i) pair. A router scores experts,
    selects top-k, and computes a weighted sum of their outputs.

    Note: this loop implementation materializes a (batch, seq, K, out_features)
    stack tensor whose memory scales with K. For K >= 16 it becomes the memory
    bottleneck. The RoutedLoRA class (added in later commits) uses a shared
    bottleneck whose memory cost is constant in K*r.
    """

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
            nn.init.kaiming_uniform_(A.weight)
        for B in self.experts_B:
            nn.init.zeros_(B.weight)

    def forward(self, x):
        x_dropped = self.dropout(x)

        scores = self.router(x)
        top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=-1)
        self._last_routing_indices = top_k_indices.detach()
        top_k_weights = F.softmax(top_k_scores, dim=-1)

        expert_outputs = torch.stack(
            [B(A(x_dropped)) for A, B in zip(self.experts_A, self.experts_B)],
            dim=-2,
        )

        idx = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, expert_outputs.shape[-1])
        selected = torch.gather(expert_outputs, dim=-2, index=idx)
        output = (selected * top_k_weights.unsqueeze(-1)).sum(dim=-2)
        return output * self.scaling


class TMLoRA(nn.Module):
    """Token-Modulated LoRA: shared A/B with additive expert vector modulation.

    Forward: output = GELU(x.A + E_token) . B, where E_token is a top-k
    weighted sum of learned expert vectors.
    """

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

        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        x_dropped = self.dropout(x)

        scores = self.router(x)
        top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_scores, dim=-1)

        selected = self.expert_vectors[top_k_indices]
        E_token = (selected * top_k_weights.unsqueeze(-1)).sum(dim=-2)

        hidden = self.A(x_dropped) + E_token
        return self.B(F.gelu(hidden)) * self.scaling


# ---------------------------------------------------------------------------
# Routed LoRA (unified shared-bottleneck architecture)
# ---------------------------------------------------------------------------

class RoutedLoRA(nn.Module):
    """Routed LoRA adapter with shared A/B matrices and top-k expert routing.

    Six router parameterizations are supported via `router_type`:

    Baseline routers (Phase B):
      - "linear":  scores = Linear(d, K)(x). Cost: O(d*K).
      - "lowrank": scores = (W_query(x)) @ keys.T, W_query: d->rdim,
                   keys: (K, rdim). Cost: O(d*rdim + K*rdim).

    Efficient-routing variants (Phase D router comparison study):
      - "cosine":  same as lowrank but with L2 normalization on q and keys
                   before the dot product. Same cost as lowrank.
      - "hierarchical": two-level sqrt(K) routing. router_l1: d->G groups;
                   router_l2: d->K/G within-group experts (shared across
                   selected groups). Pick top-g groups, top-(top_k/g) experts
                   per group. Cost: O(d*sqrt(K)). Requires K and top_k square.
      - "product_key": Lample et al. 2019. Two scorers (d -> sqrt(K) each).
                   Score for expert (i,j) = scorer1[i] + scorer2[j]. Top-k
                   over the K=sqrt(K)*sqrt(K) product space. Cost: O(d*sqrt(K)).
                   Any expert is reachable (no group bottleneck).
      - "early_shared": one full linear router (Linear(d, K)) computes routing
                   from the FIRST RoutedLoRA layer's input; all subsequent
                   RoutedLoRA modules in the model reuse the same top-k
                   indices and weights. The owner is designated externally
                   (build_model in src/model.py); non-owners have their
                   router deleted. Cost amortized across L layers: O(d*K / L).
    """

    # Class-level cache for early_shared routing. Owner writes; non-owners read.
    # Overwritten at every owner forward pass.
    _shared_routing_cache = {"indices": None, "weights": None}

    def __init__(self, in_features, out_features, rank=1, alpha=32, dropout=0.0,
                 num_experts=64, top_k=16, router_type="lowrank", router_dim=16,
                 **kwargs):
        super().__init__()
        assert router_type in (
            "linear", "lowrank", "cosine", "hierarchical", "product_key", "early_shared"
        ), f"unknown router_type {router_type!r}"
        self.num_experts = num_experts
        self.rank = rank
        self.top_k = top_k
        self.router_type = router_type
        self.scaling = alpha / top_k
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        bottleneck = num_experts * rank
        self.A = nn.Linear(in_features, bottleneck, bias=False)
        self.B = nn.Linear(bottleneck, out_features, bias=False)

        # Default ownership flag for early_shared. build_model resets this
        # to False on all but the first early_shared module.
        self.is_router_owner = True

        if router_type in ("linear", "early_shared"):
            self.router = nn.Linear(in_features, num_experts, bias=False)
        elif router_type in ("lowrank", "cosine"):
            self.W_query = nn.Linear(in_features, router_dim, bias=False)
            self.keys = nn.Parameter(torch.randn(num_experts, router_dim) * 0.01)
        elif router_type == "hierarchical":
            G = int(num_experts ** 0.5)
            assert G * G == num_experts, \
                f"hierarchical requires square num_experts, got {num_experts}"
            g_active = int(top_k ** 0.5)
            assert g_active * g_active == top_k, \
                f"hierarchical requires square top_k, got {top_k}"
            self.G = G
            self.K_per_g = num_experts // G
            self.g_active = g_active
            self.k_per_g_active = top_k // g_active
            self.router_l1 = nn.Linear(in_features, G, bias=False)
            self.router_l2 = nn.Linear(in_features, self.K_per_g, bias=False)
        elif router_type == "product_key":
            sqrt_K = int(num_experts ** 0.5)
            assert sqrt_K * sqrt_K == num_experts, \
                f"product_key requires square num_experts, got {num_experts}"
            self.sqrt_K = sqrt_K
            self.scorer1 = nn.Linear(in_features, sqrt_K, bias=False)
            self.scorer2 = nn.Linear(in_features, sqrt_K, bias=False)

        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)

    def _route(self, x):
        """Compute top-k indices and softmax weights for the given input.

        Returns (top_k_indices, top_k_weights), both shape (batch, seq, top_k).
        """
        if self.router_type == "linear":
            scores = self.router(x)
            top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=-1)
            return top_k_indices, F.softmax(top_k_scores, dim=-1)

        if self.router_type == "lowrank":
            q = self.W_query(x)
            scores = torch.matmul(q, self.keys.t())
            top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=-1)
            return top_k_indices, F.softmax(top_k_scores, dim=-1)

        if self.router_type == "cosine":
            q = F.normalize(self.W_query(x), dim=-1)
            k = F.normalize(self.keys, dim=-1)
            scores = torch.matmul(q, k.t())
            top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=-1)
            return top_k_indices, F.softmax(top_k_scores, dim=-1)

        if self.router_type == "product_key":
            s1 = self.scorer1(x)
            s2 = self.scorer2(x)
            scores = (s1.unsqueeze(-1) + s2.unsqueeze(-2)).flatten(start_dim=-2)
            top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=-1)
            return top_k_indices, F.softmax(top_k_scores, dim=-1)

        if self.router_type == "hierarchical":
            g_scores = self.router_l1(x)
            top_g_scores, top_g_idx = torch.topk(g_scores, self.g_active, dim=-1)
            e_scores = self.router_l2(x)
            top_e_scores, top_e_idx = torch.topk(e_scores, self.k_per_g_active, dim=-1)
            g_idx_exp = top_g_idx.unsqueeze(-1).expand(
                *top_g_idx.shape, self.k_per_g_active)
            e_idx_exp = top_e_idx.unsqueeze(-2).expand(
                *top_e_idx.shape[:-1], self.g_active, self.k_per_g_active)
            global_idx = g_idx_exp * self.K_per_g + e_idx_exp
            top_k_indices = global_idx.flatten(start_dim=-2)
            g_scores_exp = top_g_scores.unsqueeze(-1).expand(
                *top_g_scores.shape, self.k_per_g_active)
            e_scores_exp = top_e_scores.unsqueeze(-2).expand(
                *top_e_scores.shape[:-1], self.g_active, self.k_per_g_active)
            combined = (g_scores_exp + e_scores_exp).flatten(start_dim=-2)
            return top_k_indices, F.softmax(combined, dim=-1)

        # early_shared: only the owner runs the router; non-owners read cache.
        if self.is_router_owner:
            scores = self.router(x)
            top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=-1)
            top_k_weights = F.softmax(top_k_scores, dim=-1)
            RoutedLoRA._shared_routing_cache["indices"] = top_k_indices
            RoutedLoRA._shared_routing_cache["weights"] = top_k_weights
            return top_k_indices, top_k_weights
        return (RoutedLoRA._shared_routing_cache["indices"],
                RoutedLoRA._shared_routing_cache["weights"])

    def forward(self, x):
        x_dropped = self.dropout(x)

        # Bottleneck: (batch, seq, K*r) -> view as (batch, seq, K, r)
        z = self.A(x_dropped)
        z = z.view(*z.shape[:-1], self.num_experts, self.rank)

        top_k_indices, top_k_weights = self._route(x_dropped)
        self._last_routing_indices = top_k_indices.detach()

        # Build (batch, seq, K) gate tensor — zero outside top-k positions
        gate = torch.zeros(
            *x_dropped.shape[:-1], self.num_experts,
            device=x_dropped.device, dtype=top_k_weights.dtype,
        )
        gate.scatter_(-1, top_k_indices, top_k_weights)

        # Apply gate per-expert: (batch, seq, K, r) * (batch, seq, K, 1)
        z_gated = z * gate.unsqueeze(-1)
        z_flat = z_gated.flatten(start_dim=-2)
        return self.B(z_flat) * self.scaling


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class LinearWithLoRA(nn.Module):
    """Wraps a frozen nn.Linear with an additive LoRA branch.

    Computes: output = original(x) + lora(x.float()).to(base_dtype)

    The LoRA branch is run in FP32 for training stability while the base
    layer stays in the model's native precision (typically FP16).
    """

    def __init__(self, original_linear, lora_module):
        super().__init__()
        self.original_linear = original_linear
        self.lora = lora_module

    def forward(self, x):
        base_out = self.original_linear(x)
        lora_out = self.lora(x.float()).to(base_out.dtype)
        return base_out + lora_out
