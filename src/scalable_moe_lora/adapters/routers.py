"""Routers for routed LoRA adapters.

Every router takes a hidden state x in R^d and returns (topk_idx, topk_weights):
  - topk_idx:     long tensor, shape (..., top_k) — indices of selected experts in [0, K)
  - topk_weights: float tensor, shape (..., top_k) — softmax-normalized gates summing to 1

Both MoELoRA and MoELoRA delegate to these through `build_router`.

Router catalogue:
  - linear            Linear(d, K), O(d*K) params
  - lowrank           W_query (d, r_R) + keys (K, r_R), O(d*r_R + K*r_R) params
  - cosine            lowrank w/ L2 normalization on q and keys before dot product
  - hierarchical      two-level sqrt(K) routing; O(d*sqrt(K)) params
  - product_key       Lample et al. 2019 factored scoring; O(d*sqrt(K)) params
  - product_key_temp  product_key + learnable per-module temperature on the
                      top-k softmax. Aux loss `p_i` uses the raw scores so the
                      load-balance penalty doesn't push tau back to 1. Tests
                      whether the cheap router's near-uniform soft gates can
                      sharpen given a free temperature DOF.
  - multihead_pk      H parallel product_key heads; scores averaged. O(H*d*sqrt(K)).
                      H=4 at K=64 matches a linear router's parameter count but is
                      structurally factored (every expert reachable).
  - two_stage_pk      product_key for top-k selection + a separate rank-r_g gate
                      calibration head (W_g: d->r_g, G: K x r_g) for soft-gate
                      weights. O(d*sqrt(K) + d*r_g + K*r_g). Aimed at the gate-fidelity
                      hypothesis that linear's OOD edge is gate-calibration capacity.
  - early_shared      one Linear(d, K) at the first LoRA layer; indices cached and reused
                      at all downstream injection points. Cost amortized across layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_router_init(linear_or_param, kind):
    """Optional router init override.
      - 'default'      : leave PyTorch / caller-set init alone.
      - 'kaiming'      : kaiming-uniform on the weight (or .weight if Linear).
      - 'small_randn'  : N(0, 0.01) — same scale as LowRankRouter.keys default.
    """
    w = linear_or_param.weight if hasattr(linear_or_param, "weight") else linear_or_param
    if kind == "default":
        return
    if kind == "kaiming":
        nn.init.kaiming_uniform_(w, a=5 ** 0.5)
        return
    if kind == "small_randn":
        nn.init.normal_(w, mean=0.0, std=0.01)
        return
    raise ValueError(f"unknown router_init {kind!r}")


class LinearRouter(nn.Module):
    """scores = Linear(d, K)(x), then top-k + softmax.

    `router_init` (default 'default'): override the bare Linear init. PyTorch's
    default is kaiming-uniform with a=sqrt(5); 'small_randn' applies
    N(0, 0.01) — useful as an ablation on whether linear's gate-fidelity edge
    is initialization-driven.
    """

    def __init__(self, d, num_experts, top_k, router_init="default", **kwargs):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(d, num_experts, bias=False)
        _apply_router_init(self.router, router_init)

    def forward(self, x):
        scores = self.router(x)
        topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        return topk_idx, F.softmax(topk_vals, dim=-1), scores


class LowRankRouter(nn.Module):
    """scores = (W_query(x)) @ keys^T, where W_query: d -> r_R and keys: (K, r_R).

    `router_init` (default 'default') overrides W_query's init only; `keys` keep
    their original N(0, 0.01) init. 'kaiming' tests whether lowrank's
    coarser-than-linear gate-fidelity is partly an init-scale gap.
    """

    def __init__(self, d, num_experts, top_k, router_dim=16, router_init="default", **kwargs):
        super().__init__()
        self.top_k = top_k
        self.W_query = nn.Linear(d, router_dim, bias=False)
        self.keys = nn.Parameter(torch.randn(num_experts, router_dim) * 0.01)
        _apply_router_init(self.W_query, router_init)

    def forward(self, x):
        q = self.W_query(x)
        scores = torch.matmul(q, self.keys.t())
        topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        return topk_idx, F.softmax(topk_vals, dim=-1), scores


class CosineRouter(nn.Module):
    """Lowrank with L2 normalization on query and keys before the dot product."""

    def __init__(self, d, num_experts, top_k, router_dim=16, **kwargs):
        super().__init__()
        self.top_k = top_k
        self.W_query = nn.Linear(d, router_dim, bias=False)
        self.keys = nn.Parameter(torch.randn(num_experts, router_dim) * 0.01)

    def forward(self, x):
        q = F.normalize(self.W_query(x), dim=-1)
        k = F.normalize(self.keys, dim=-1)
        scores = torch.matmul(q, k.t())
        topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        return topk_idx, F.softmax(topk_vals, dim=-1), scores


class HierarchicalRouter(nn.Module):
    """Two-level sqrt(K) routing: pick top-g groups, then top-(k/g) experts within each."""

    def __init__(self, d, num_experts, top_k, **kwargs):
        super().__init__()
        G = int(num_experts ** 0.5)
        assert G * G == num_experts, f"hierarchical requires square num_experts, got {num_experts}"
        g_active = int(top_k ** 0.5)
        assert g_active * g_active == top_k, f"hierarchical requires square top_k, got {top_k}"
        self.top_k = top_k
        self.G = G
        self.K_per_g = num_experts // G
        self.g_active = g_active
        self.k_per_g_active = top_k // g_active
        self.router_l1 = nn.Linear(d, G, bias=False)
        self.router_l2 = nn.Linear(d, self.K_per_g, bias=False)

    def forward(self, x):
        g_scores = self.router_l1(x)                              # (..., G)
        top_g_scores, top_g_idx = torch.topk(g_scores, self.g_active, dim=-1)
        e_scores = self.router_l2(x)                              # (..., K/G)
        top_e_scores, top_e_idx = torch.topk(e_scores, self.k_per_g_active, dim=-1)
        g_idx_exp = top_g_idx.unsqueeze(-1).expand(*top_g_idx.shape, self.k_per_g_active)
        e_idx_exp = top_e_idx.unsqueeze(-2).expand(*top_e_idx.shape[:-1],
                                                   self.g_active, self.k_per_g_active)
        global_idx = g_idx_exp * self.K_per_g + e_idx_exp          # (..., g_a, k_per_g_a)
        topk_idx = global_idx.flatten(start_dim=-2)                # (..., top_k)
        g_exp = top_g_scores.unsqueeze(-1).expand(*top_g_scores.shape, self.k_per_g_active)
        e_exp = top_e_scores.unsqueeze(-2).expand(*top_e_scores.shape[:-1],
                                                  self.g_active, self.k_per_g_active)
        combined = (g_exp + e_exp).flatten(start_dim=-2)
        return topk_idx, F.softmax(combined, dim=-1), None  # full K-wide scores not cheap here


class ProductKeyRouter(nn.Module):
    """Lample et al. 2019: two scorers (d -> sqrt(K)); expert (i,j) scores s1[i] + s2[j]."""

    def __init__(self, d, num_experts, top_k, **kwargs):
        super().__init__()
        sqrt_K = int(num_experts ** 0.5)
        assert sqrt_K * sqrt_K == num_experts, f"product_key requires square num_experts, got {num_experts}"
        self.top_k = top_k
        self.sqrt_K = sqrt_K
        self.scorer1 = nn.Linear(d, sqrt_K, bias=False)
        self.scorer2 = nn.Linear(d, sqrt_K, bias=False)

    def forward(self, x):
        s1 = self.scorer1(x)                                       # (..., sqrt_K)
        s2 = self.scorer2(x)
        scores = (s1.unsqueeze(-1) + s2.unsqueeze(-2)).flatten(start_dim=-2)  # (..., K)
        topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        return topk_idx, F.softmax(topk_vals, dim=-1), scores


class TempProductKeyRouter(nn.Module):
    """Product-key router with a learnable per-module temperature on the top-k
    softmax. Selection (top-k indices) is scale-invariant and full_scores
    returned to the trainer is the *raw* product-key score, so the aux-loss
    `p_i = softmax(full_scores).mean(0)` continues to prefer balanced load
    without being pulled by the temperature parameter.

    Parameterized as tau = exp(log_tau) for positivity; log_tau = 0 at init
    reproduces plain product_key. If tau shrinks during training, gates of the
    selected experts grow sharper (max gate value increases, normalized
    entropy drops). The gate-fidelity hypothesis predicts that letting tau move freely
    transfers some of linear's gate-fidelity advantage into the cheap router.

    One scalar (log_tau) per MoELoRA injection — 32 extra params total at
    target_modules=qv on a 16-layer model.
    """

    def __init__(self, d, num_experts, top_k, **kwargs):
        super().__init__()
        sqrt_K = int(num_experts ** 0.5)
        assert sqrt_K * sqrt_K == num_experts, (
            f"product_key_temp requires square num_experts, got {num_experts}"
        )
        self.top_k = top_k
        self.sqrt_K = sqrt_K
        self.scorer1 = nn.Linear(d, sqrt_K, bias=False)
        self.scorer2 = nn.Linear(d, sqrt_K, bias=False)
        self.log_temperature = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        s1 = self.scorer1(x)
        s2 = self.scorer2(x)
        scores = (s1.unsqueeze(-1) + s2.unsqueeze(-2)).flatten(start_dim=-2)
        topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        tau = torch.exp(self.log_temperature)
        return topk_idx, F.softmax(topk_vals / tau, dim=-1), scores


class MultiHeadProductKeyRouter(nn.Module):
    """H parallel product-key heads; per-head K-wide scores averaged before top-k.

    Vectorized: a single Linear(d, H*sqrt_K) per scorer instead of H Linear(d, sqrt_K)
    in a ModuleList. The (..., H*sqrt_K) output is reshaped to (..., H, sqrt_K) and
    the per-head outer-sum + average over H is one batched op. One kernel launch per
    scorer instead of H — at H=4, K=64 over 32 MoELoRA modules this removed a
    ~40 % per-step slowdown versus plain product_key (job 3162385 timed out at the
    24 h wall on the loop version).
    """

    def __init__(self, d, num_experts, top_k, num_heads=4, **kwargs):
        super().__init__()
        sqrt_K = int(num_experts ** 0.5)
        assert sqrt_K * sqrt_K == num_experts, f"multihead_pk requires square num_experts, got {num_experts}"
        self.top_k = top_k
        self.sqrt_K = sqrt_K
        self.num_heads = num_heads
        self.scorer1 = nn.Linear(d, num_heads * sqrt_K, bias=False)
        self.scorer2 = nn.Linear(d, num_heads * sqrt_K, bias=False)

    def forward(self, x):
        H, sK = self.num_heads, self.sqrt_K
        s1 = self.scorer1(x).view(*x.shape[:-1], H, sK)                           # (..., H, sK)
        s2 = self.scorer2(x).view(*x.shape[:-1], H, sK)                           # (..., H, sK)
        head_scores = (s1.unsqueeze(-1) + s2.unsqueeze(-2)).flatten(start_dim=-2)  # (..., H, K)
        scores = head_scores.mean(-2)                                              # (..., K)
        topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        return topk_idx, F.softmax(topk_vals, dim=-1), scores


class TwoStagePKRouter(nn.Module):
    """Stage 1: product-key picks top-k indices.
    Stage 2: a rank-r_g gate head (W_g: d->r_g, G: (K, r_g)) calibrates soft-gate weights
             for the selected indices via dot product q_g . G[idx].

    The returned topk_weights are softmax over (select_topk + gate_topk) so that *both*
    heads receive task-loss gradient: the selection scorer through its own top-k logits,
    the gate head through its calibration logits. Without the additive coupling, the
    selection scorer would only see the 0.01-coefficient aux-loss gradient and never
    actually learn what to select.

    full_scores returned to the trainer is the product-key K-wide selection score, used
    only by the aux-loss p_i term in MoELoRA.forward.
    """

    def __init__(self, d, num_experts, top_k, gate_rank=16, **kwargs):
        super().__init__()
        sqrt_K = int(num_experts ** 0.5)
        assert sqrt_K * sqrt_K == num_experts, f"two_stage_pk requires square num_experts, got {num_experts}"
        self.top_k = top_k
        self.sqrt_K = sqrt_K
        self.scorer1 = nn.Linear(d, sqrt_K, bias=False)
        self.scorer2 = nn.Linear(d, sqrt_K, bias=False)
        self.W_gate = nn.Linear(d, gate_rank, bias=False)
        self.gate_keys = nn.Parameter(torch.randn(num_experts, gate_rank) * 0.01)

    def forward(self, x):
        s1 = self.scorer1(x)
        s2 = self.scorer2(x)
        select_scores = (s1.unsqueeze(-1) + s2.unsqueeze(-2)).flatten(start_dim=-2)  # (..., K)
        select_topk_vals, topk_idx = torch.topk(select_scores, self.top_k, dim=-1)

        q_g = self.W_gate(x)                                          # (..., r_g)
        gate_full = torch.matmul(q_g, self.gate_keys.t())             # (..., K)
        gate_topk = torch.gather(gate_full, -1, topk_idx)              # (..., top_k)

        combined = select_topk_vals + gate_topk                        # (..., top_k)
        return topk_idx, F.softmax(combined, dim=-1), select_scores


class EarlySharedRouter(nn.Module):
    """Single Linear(d, K) at the first LoRA layer; all later layers read the owner's cache.

    Owner is set externally in `build_model._set_early_shared_owner`:
      - the first EarlySharedRouter in iteration order gets `is_owner=True`; it
        computes routing and stores it in `self._cache` on each forward.
      - all other EarlySharedRouter instances get `is_owner=False` and hold a
        reference to the owner in `_owner_ref` (set via `object.__setattr__` to
        bypass nn.Module's submodule registration, avoiding double-counting the
        owner's parameters). Their forward just reads `self._owner_ref._cache`.

    Why instance-level, not class-level: with `use_reentrant=False` gradient
    checkpointing, backward-pass recomputes can interleave across layers, and a
    class-level cache would be overwritten out-of-order during backward.
    """

    def __init__(self, d, num_experts, top_k, **kwargs):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.router = nn.Linear(d, num_experts, bias=False)
        self.is_owner = True
        self._cache = None

    def forward(self, x):
        if self.is_owner:
            scores = self.router(x)
            topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=-1)
            weights = F.softmax(topk_vals, dim=-1)
            self._cache = (topk_idx, weights, scores)
            return topk_idx, weights, scores
        return self._owner_ref._cache


_REGISTRY = {
    "linear": LinearRouter,
    "lowrank": LowRankRouter,
    "cosine": CosineRouter,
    "hierarchical": HierarchicalRouter,
    "product_key": ProductKeyRouter,
    "product_key_temp": TempProductKeyRouter,
    "multihead_pk": MultiHeadProductKeyRouter,
    "two_stage_pk": TwoStagePKRouter,
    "early_shared": EarlySharedRouter,
}


def build_router(kind, d, num_experts, top_k, **kwargs):
    """Return a Router module of the requested kind.

    Router forward: x (..., d) -> (topk_idx, topk_weights, full_scores_or_None)
    """
    if kind not in _REGISTRY:
        raise ValueError(f"unknown router_type {kind!r}; available: {list(_REGISTRY)}")
    return _REGISTRY[kind](d=d, num_experts=num_experts, top_k=top_k, **kwargs)
