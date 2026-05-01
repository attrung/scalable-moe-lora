"""Sanity tests for the router catalogue and MoELoRA forward/backward."""

import math

import pytest
import torch
import torch.nn as nn

from scalable_moe_lora.adapters import MoELoRA, build_router


ROUTERS_K64 = [
    "linear", "lowrank", "cosine", "hierarchical",
    "product_key", "multihead_pk", "two_stage_pk",
    "product_key_temp", "early_shared",
]


@pytest.mark.parametrize("router_type", ROUTERS_K64)
def test_router_shapes(router_type):
    """Every router returns (top-k indices, top-k weights summing to 1, optional K-wide scores)."""
    torch.manual_seed(0)
    d, K, k = 64, 64, 16
    r = build_router(router_type, d=d, num_experts=K, top_k=k)
    x = torch.randn(2, 8, d)
    idx, w, full = r(x)

    assert idx.shape == (2, 8, k), f"idx shape {idx.shape}"
    assert w.shape   == (2, 8, k), f"w shape {w.shape}"
    assert torch.allclose(w.sum(-1), torch.ones(2, 8), atol=1e-4), \
        "top-k weights must sum to 1 per token"
    if full is not None:
        assert full.shape == (2, 8, K), f"full_scores shape {full.shape}"


def test_multihead_pk_vectorized_eq_loop():
    """The vectorized MultiHeadProductKeyRouter must be bit-exact with a manual H-loop
    over per-head Linear modules built from sliced weights."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    d, K, k, H = 64, 64, 16, 4
    sK = int(math.sqrt(K))

    new = build_router("multihead_pk", d=d, num_experts=K, top_k=k, num_heads=H)

    # Build a manual loop equivalent by slicing new's stacked weight into H per-head Linears.
    s1m = nn.ModuleList([nn.Linear(d, sK, bias=False) for _ in range(H)])
    s2m = nn.ModuleList([nn.Linear(d, sK, bias=False) for _ in range(H)])
    with torch.no_grad():
        for h in range(H):
            s1m[h].weight.copy_(new.scorer1.weight[h*sK:(h+1)*sK])
            s2m[h].weight.copy_(new.scorer2.weight[h*sK:(h+1)*sK])

    x = torch.randn(2, 8, d)
    idx_n, w_n, scores_n = new(x)

    scores_sum = None
    for s1l, s2l in zip(s1m, s2m):
        s1, s2 = s1l(x), s2l(x)
        head_scores = (s1.unsqueeze(-1) + s2.unsqueeze(-2)).flatten(start_dim=-2)
        scores_sum = head_scores if scores_sum is None else scores_sum + head_scores
    scores_loop = scores_sum / H

    assert torch.allclose(scores_n, scores_loop, atol=1e-6)
    assert torch.equal(idx_n, torch.topk(scores_loop, k, dim=-1).indices)


def test_two_stage_pk_grad_flow():
    """Both selection scorers (scorer1, scorer2) and gate head (W_gate, gate_keys)
    must receive task-loss gradient. A previous bug routed gradient only through
    the gate head."""
    torch.manual_seed(0)
    d, K, k = 64, 64, 16
    r = build_router("two_stage_pk", d=d, num_experts=K, top_k=k, gate_rank=16)
    x = torch.randn(2, 8, d, requires_grad=True)
    idx, w, _ = r(x)
    w.sum().backward()
    for n, p in r.named_parameters():
        assert p.grad is not None and float(p.grad.norm()) > 0, \
            f"{n} did not receive gradient"


def test_routed_lora_forward_backward():
    """MoELoRA forward + backward runs without shape errors and emits aux-loss."""
    torch.manual_seed(0)
    d, K, k = 64, 64, 16
    layer = MoELoRA(d, d, rank=1, alpha=32, num_experts=K, top_k=k, router_type="product_key")
    # Default B init is zeros (LoRA convention); perturb so y is non-trivial for grad flow.
    nn.init.normal_(layer.B.weight, std=0.01)
    x = torch.randn(2, 8, d)
    y = layer(x)
    assert y.shape == (2, 8, d)
    assert layer._last_aux_loss is not None and layer._last_aux_loss.item() > 0
    (y.pow(2).sum() + 0.01 * layer._last_aux_loss).backward()
    assert layer.A.weight.grad is not None
    assert layer.B.weight.grad is not None


def test_early_shared_aux_loss_owner_only():
    """EarlySharedRouter followers reuse the owner's routing decision verbatim,
    so they must NOT contribute aux-loss (otherwise the effective coefficient
    is multiplied by the number of injection sites). After the fix, summed
    aux loss across N MoELoRAs sharing one owner equals the owner's aux loss."""
    from scalable_moe_lora.adapters import collect_aux_loss
    torch.manual_seed(0)
    d, K, k, N = 64, 64, 16, 4

    # Build N MoELoRA modules wired the way build_model wires real layers:
    # the first router instance is the owner; the rest follow it.
    layers = nn.ModuleList([
        MoELoRA(d, d, rank=1, alpha=32, num_experts=K, top_k=k, router_type="early_shared")
        for _ in range(N)
    ])
    owner = layers[0].router
    for i in range(1, N):
        layers[i].router.is_owner = False
        object.__setattr__(layers[i].router, "_owner_ref", owner)

    x = torch.randn(2, 8, d)
    for layer in layers:
        layer(x)

    owner_aux = layers[0]._last_aux_loss
    follower_aux = [layers[i]._last_aux_loss for i in range(1, N)]
    assert owner_aux is not None
    assert all(a is None for a in follower_aux), "followers must emit None aux loss"

    summed = collect_aux_loss(layers).item()
    assert abs(summed - owner_aux.item()) < 1e-6, (
        f"early_shared total aux ({summed}) must equal owner-only ({owner_aux.item()})"
    )


def test_temperature_router_invariants():
    """product_key_temp: tau scales gate magnitudes but leaves top-k selection
    and full_scores invariant."""
    torch.manual_seed(0)
    d, K, k = 64, 64, 16
    r1 = build_router("product_key_temp", d=d, num_experts=K, top_k=k)
    r2 = build_router("product_key_temp", d=d, num_experts=K, top_k=k)
    r2.scorer1.load_state_dict(r1.scorer1.state_dict())
    r2.scorer2.load_state_dict(r1.scorer2.state_dict())
    with torch.no_grad():
        r2.log_temperature.fill_(-2.0)  # tau ~ 0.135 → much sharper

    x = torch.randn(2, 8, d)
    idx1, w1, full1 = r1(x)
    idx2, w2, full2 = r2(x)
    assert torch.equal(idx1, idx2), "selection must be invariant to tau"
    assert torch.allclose(full1, full2, atol=1e-6), "full_scores must be invariant to tau"
    # Lower tau → sharper gates: max gate value should be larger.
    assert w2.max(-1).values.mean() > w1.max(-1).values.mean()
