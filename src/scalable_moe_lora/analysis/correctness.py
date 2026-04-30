"""Numerical-equivalence tests for the DispatchMoELoRA multi-router implementation.

Test 1: Dispatch (linear router) vs stack-and-gather reference at matched weights.
  Both must produce identical outputs (to float tol) at init.
  Covers K ∈ {4, 8, 16, 32, 64} with matched r, top_k.

Test 2: Dispatch with each of the 6 routers (linear / lowrank / cosine /
  hierarchical / product_key / early_shared) must forward without error and
  return a finite, differentiable output.

Run:
    cd <repo-root>
    source env.sh
    python3 scripts/correctness_test.py
"""

import os
import sys


import torch
import torch.nn as nn
import torch.nn.functional as F

from scalable_moe_lora.adapters import DispatchMoELoRA, RoutedLoRA


class ReferenceMoELoRA(nn.Module):
    """Naive K-expert top-k MoE-LoRA (stack-and-gather). Reference for equivalence."""

    def __init__(self, d, K, r, top_k):
        super().__init__()
        self.K = K
        self.top_k = top_k
        self.A = nn.ParameterList([nn.Parameter(torch.empty(r, d)) for _ in range(K)])
        self.B = nn.ParameterList([nn.Parameter(torch.empty(d, r)) for _ in range(K)])
        self.router = nn.Linear(d, K, bias=False)

    def forward(self, x, alpha_over_r):
        scores = self.router(x)
        topk_vals, topk_idx = scores.topk(self.top_k, dim=-1)
        gates = F.softmax(topk_vals, dim=-1)
        expert_outs = [F.linear(F.linear(x, self.A[i]), self.B[i]) for i in range(self.K)]
        Y = torch.stack(expert_outs, dim=-2)
        idx = topk_idx.unsqueeze(-1).expand(-1, -1, -1, Y.shape[-1])
        selected = torch.gather(Y, dim=-2, index=idx)
        out = (selected * gates.unsqueeze(-1)).sum(dim=-2)
        return out * alpha_over_r


def copy_weights_linear(dispatch, ref):
    """Copy stacked params + linear router from DispatchMoELoRA into ReferenceMoELoRA."""
    with torch.no_grad():
        for i in range(dispatch.num_experts):
            ref.A[i].copy_(dispatch.A[i])
            ref.B[i].copy_(dispatch.B[i])
        ref.router.weight.copy_(dispatch.router.router.weight)


def equiv_case(d, K, r, top_k, B, S, rtol=1e-4, atol=1e-4):
    torch.manual_seed(0)
    alpha = 32
    dispatch = DispatchMoELoRA(
        d, d, rank=r, alpha=alpha, dropout=0.0,
        num_experts=K, top_k=top_k, router_type="linear",
    )
    ref = ReferenceMoELoRA(d, K, r, top_k)
    copy_weights_linear(dispatch, ref)

    x = torch.randn(B, S, d, dtype=torch.float32)
    y_d = dispatch(x)
    y_r = ref(x, alpha / r)
    ok_fwd = torch.allclose(y_d, y_r, rtol=rtol, atol=atol)
    max_err = (y_d - y_r).abs().max().item()

    x.requires_grad_(True)
    loss_d = dispatch(x).sum()
    gA_d = torch.autograd.grad(loss_d, dispatch.A, retain_graph=True)[0]
    gB_d = torch.autograd.grad(loss_d, dispatch.B, retain_graph=True)[0]
    gR_d = torch.autograd.grad(loss_d, dispatch.router.router.weight)[0]

    loss_r = ref(x, alpha / r).sum()
    gA_r = torch.stack(list(torch.autograd.grad(loss_r, list(ref.A), retain_graph=True)), dim=0)
    gB_r = torch.stack(list(torch.autograd.grad(loss_r, list(ref.B), retain_graph=True)), dim=0)
    gR_r = torch.autograd.grad(loss_r, ref.router.weight)[0]

    ok_gA = torch.allclose(gA_d, gA_r, rtol=rtol, atol=atol)
    ok_gB = torch.allclose(gB_d, gB_r, rtol=rtol, atol=atol)
    ok_gR = torch.allclose(gR_d, gR_r, rtol=rtol, atol=atol)

    tag = f"d={d} K={K} r={r} k={top_k} B={B} S={S}"
    ok = ok_fwd and ok_gA and ok_gB and ok_gR
    print(f"[EQUIV {tag}] fwd={max_err:.2e} gA={ok_gA} gB={ok_gB} gR={ok_gR} "
          f"=> {'PASS' if ok else 'FAIL'}")
    return ok


def router_smoke_case(router_type, d=256, K=64, r=1, top_k=16, B=2, S=16, **kwargs):
    """Verify DispatchMoELoRA forwards + backwards cleanly with the given router."""
    torch.manual_seed(0)
    dispatch = DispatchMoELoRA(
        d, d, rank=r, alpha=32, dropout=0.0,
        num_experts=K, top_k=top_k, router_type=router_type, **kwargs,
    )
    x = torch.randn(B, S, d, dtype=torch.float32, requires_grad=True)
    y = dispatch(x)
    ok_shape = y.shape == (B, S, d)
    ok_finite = bool(torch.isfinite(y).all())
    # Backprop to x (and to router params) works:
    loss = y.sum() + dispatch._last_aux_loss
    loss.backward()
    ok_bwd = (x.grad is not None) and bool(torch.isfinite(x.grad).all())
    aux = dispatch._last_aux_loss.item()
    ok = ok_shape and ok_finite and ok_bwd
    print(f"[ROUTER {router_type:14s}] shape={ok_shape} finite={ok_finite} "
          f"bwd={ok_bwd} aux={aux:.3f} => {'PASS' if ok else 'FAIL'}")
    return ok


def routed_vs_dispatch_equiv_case(d, K, r, top_k, B, S, rtol=1e-4, atol=1e-4):
    """Head-to-head numerical equivalence: RoutedLoRA ≡ DispatchMoELoRA at matched
    weights, matched scale (α/top_k on both), matched linear router. Verifies:
      - forward outputs match
      - aux_loss values match
      - ∂L/∂A (reshaped), ∂L/∂B (reshaped), ∂L/∂router_weight match
    If this passes at multiple shapes, the two classes are the same model up to
    implementation-level float rounding.
    """
    torch.manual_seed(0)
    alpha = 32
    routed = RoutedLoRA(d, d, rank=r, alpha=alpha, dropout=0.0,
                        num_experts=K, top_k=top_k, router_type="linear")
    dispatch = DispatchMoELoRA(d, d, rank=r, alpha=alpha, dropout=0.0,
                               num_experts=K, top_k=top_k, router_type="linear")

    # Override dispatch scaling to match routed (α/top_k). This removes the
    # only non-numerical difference between the two classes.
    dispatch.scaling = routed.scaling

    # Copy weights routed → dispatch, mapping block-of-shared-matrix to
    # expert-indexed independent matrices.
    with torch.no_grad():
        for i in range(K):
            dispatch.A[i].copy_(routed.A.weight[i*r:(i+1)*r, :])
            dispatch.B[i].copy_(routed.B.weight[:, i*r:(i+1)*r])
        dispatch.router.router.weight.copy_(routed.router.router.weight)

    x = torch.randn(B, S, d, dtype=torch.float32)
    y_r = routed(x)
    y_d = dispatch(x)
    aux_r = routed._last_aux_loss.item()
    aux_d = dispatch._last_aux_loss.item()

    ok_fwd = torch.allclose(y_r, y_d, rtol=rtol, atol=atol)
    max_fwd_err = (y_r - y_d).abs().max().item()
    ok_aux = abs(aux_r - aux_d) < atol

    # Grad check — build fresh graphs so the two autograd.grad calls don't collide.
    loss_r = routed(x).sum() + routed._last_aux_loss
    gA_r = torch.autograd.grad(loss_r, routed.A.weight, retain_graph=True)[0]      # (Kr, d)
    gB_r = torch.autograd.grad(loss_r, routed.B.weight, retain_graph=True)[0]      # (d, Kr)
    gR_r = torch.autograd.grad(loss_r, routed.router.router.weight)[0]             # (K, d)

    loss_d = dispatch(x).sum() + dispatch._last_aux_loss
    gA_d = torch.autograd.grad(loss_d, dispatch.A, retain_graph=True)[0]           # (K, r, d)
    gB_d = torch.autograd.grad(loss_d, dispatch.B, retain_graph=True)[0]           # (K, d, r)
    gR_d = torch.autograd.grad(loss_d, dispatch.router.router.weight)[0]           # (K, d)

    # Reshape routed grads to dispatch layout: blocks along expert axis.
    gA_r_map = gA_r.view(K, r, d)
    gB_r_map = torch.stack(
        [gB_r[:, i*r:(i+1)*r] for i in range(K)], dim=0
    )                                                                               # (K, d, r)

    ok_gA = torch.allclose(gA_r_map, gA_d, rtol=rtol, atol=atol)
    ok_gB = torch.allclose(gB_r_map, gB_d, rtol=rtol, atol=atol)
    ok_gR = torch.allclose(gR_r, gR_d, rtol=rtol, atol=atol)

    max_gA_err = (gA_r_map - gA_d).abs().max().item()
    max_gB_err = (gB_r_map - gB_d).abs().max().item()
    max_gR_err = (gR_r - gR_d).abs().max().item()

    tag = f"d={d} K={K} r={r} k={top_k} B={B} S={S}"
    ok = ok_fwd and ok_aux and ok_gA and ok_gB and ok_gR
    print(f"[R↔D {tag}] fwd_err={max_fwd_err:.2e} aux_err={abs(aux_r-aux_d):.2e} "
          f"gA_err={max_gA_err:.2e} gB_err={max_gB_err:.2e} gR_err={max_gR_err:.2e} "
          f"=> {'PASS' if ok else 'FAIL'}")
    return ok


def routed_smoke_case(router_type, d=256, K=64, r=1, top_k=16, B=2, S=16, **kwargs):
    """Verify RoutedLoRA forwards + backwards cleanly with the given router."""
    torch.manual_seed(0)
    routed = RoutedLoRA(
        d, d, rank=r, alpha=32, dropout=0.0,
        num_experts=K, top_k=top_k, router_type=router_type, **kwargs,
    )
    x = torch.randn(B, S, d, dtype=torch.float32, requires_grad=True)
    y = routed(x)
    ok_shape = y.shape == (B, S, d)
    ok_finite = bool(torch.isfinite(y).all())
    aux_term = routed._last_aux_loss if routed._last_aux_loss is not None else 0.0
    loss = y.sum() + aux_term
    loss.backward()
    ok_bwd = (x.grad is not None) and bool(torch.isfinite(x.grad).all())
    aux = routed._last_aux_loss.item() if routed._last_aux_loss is not None else 0.0
    ok = ok_shape and ok_finite and ok_bwd
    print(f"[ROUTED {router_type:14s}] shape={ok_shape} finite={ok_finite} "
          f"bwd={ok_bwd} aux={aux:.3f} => {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("=== Test 1: DispatchMoELoRA (linear) vs stack-and-gather reference ===")
    equiv_cases = [
        (64,  4,  2, 1, 2,  8),
        (128, 8,  4, 2, 2, 16),
        (256, 16, 4, 4, 2, 16),
        (256, 32, 2, 8, 2, 16),
        (256, 64, 1, 16, 2, 16),
    ]
    equiv_ok = all(equiv_case(*c) for c in equiv_cases)

    print("\n=== Test 2: DispatchMoELoRA smoke — all 6 routers at K=64, r=1, k=16 ===")
    router_ok = all([
        router_smoke_case("linear"),
        router_smoke_case("lowrank", router_dim=16),
        router_smoke_case("cosine", router_dim=16),
        router_smoke_case("hierarchical"),  # requires sqrt(K) and sqrt(k) both int
        router_smoke_case("product_key"),   # requires sqrt(K) int
        router_smoke_case("early_shared"),  # single-layer smoke; owner=True by default
    ])

    print("\n=== Test 3: RoutedLoRA smoke — all 6 routers at K=64, r=1, k=16 ===")
    routed_ok = all([
        routed_smoke_case("linear"),
        routed_smoke_case("lowrank", router_dim=16),
        routed_smoke_case("cosine", router_dim=16),
        routed_smoke_case("hierarchical"),
        routed_smoke_case("product_key"),
        routed_smoke_case("early_shared"),
    ])

    print("\n=== Test 4: RoutedLoRA ↔ DispatchMoELoRA head-to-head equivalence ===")
    print("(matched weights, matched scale, matched linear router)")
    rd_equiv_ok = all([
        routed_vs_dispatch_equiv_case(d=64,  K=4,  r=2, top_k=1,  B=2, S=8),
        routed_vs_dispatch_equiv_case(d=128, K=8,  r=4, top_k=2,  B=2, S=16),
        routed_vs_dispatch_equiv_case(d=128, K=8,  r=8, top_k=2,  B=2, S=16),
        routed_vs_dispatch_equiv_case(d=256, K=16, r=4, top_k=4,  B=2, S=16),
        routed_vs_dispatch_equiv_case(d=256, K=32, r=2, top_k=8,  B=2, S=16),
        routed_vs_dispatch_equiv_case(d=256, K=64, r=1, top_k=16, B=2, S=16),
    ])

    print("")
    all_ok = equiv_ok and router_ok and routed_ok and rd_equiv_ok
    print("ALL PASS" if all_ok else "SOME CASES FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
