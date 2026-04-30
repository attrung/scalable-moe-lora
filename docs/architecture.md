# Architecture reference

Mathematical formulation of the adapter zoo and router catalogue. Implementation lives in `src/scalable_moe_lora/adapters/`.

## Notation

- `d` — hidden dimension (`d=2048` for LLaMA 3.2 1B)
- `K` — number of experts
- `r` — per-expert rank
- `top_k` — number of experts active per token (also written `k`)
- `B, S` — batch size, sequence length
- All adapters operate in fp32; the result is cast to the base model's dtype at the output sum.

## Adapter zoo

### Standard LoRA (`lora_type: standard`)

```
out(x) = W_0 x + (alpha / r) · B (A x)
A ∈ ℝ^{r × d}, B ∈ ℝ^{d × r}
trainable params per layer = 2 d r
```

### MoE-LoRA (`lora_type: moe`, `routed`, or `dispatch`)

K independent LoRA expert pairs `{(A_i, B_i)}` plus a router `R`. For each token `x` the router produces `K` scores, top-k experts are selected, and their scores are softmax-normalized into gates `{g_i}`:

```
out(x) = W_0 x + Σ_{i ∈ top-k}  g_i · (alpha / scaling) · B_i (A_i x)
trainable params per layer = 2 d K r  + router cost
active rank per token       = top_k · r
```

Three numerically-equivalent implementations (verified by `analysis/correctness.py`):

| Class            | Storage                                       | Forward                                | Activation memory |
|------------------|-----------------------------------------------|----------------------------------------|-------------------|
| `MoELoRA`        | `K × Linear(d,r)` and `K × Linear(r,d)`       | compute all K, gather top-k            | `O(B·S·K·d)` — OOMs at `K=64` on 80 GB |
| `RoutedLoRA`     | shared `A: (Kr, d)`, `B: (d, Kr)`             | one matmul, mask top-k blocks at the bottleneck | `O(B·S·K·r)` regardless of split |
| `DispatchMoELoRA`| stacked `A: (K,r,d)`, `B: (K,d,r)`            | sort-by-expert dispatch + 2 batched bmm | `O(B·S·k·d)` (top-k only) |

The production training path uses `RoutedLoRA` for its smallest-activation-footprint property (which is also why it survives `K=64` at gradient checkpointing). Dispatch is included as a reference equivalent.

### TM-LoRA (`lora_type: tm`)

A shared `A, B` pair plus a learned table `E ∈ ℝ^{K × r}` of per-expert rank-r vectors:

```
hidden(x) = A x + Σ_{i ∈ top-k}  g_i · E_i
out(x)    = W_0 x + (alpha / r) · B · GELU(hidden(x))
trainable params per layer = 2 d r + K r
active rank per token       = r  (experts modulate, they don't expand rank)
```

TM-LoRA trades expressive power for parameter efficiency: every token still passes through the same `A, B`, but the K expert vectors inject a learned per-token shift before the GELU.

## Router catalogue (`src/scalable_moe_lora/adapters/routers.py`)

All routers map `x ∈ ℝ^d` to `(top-k indices, top-k softmax weights, optional K-wide score vector)`. The K-wide vector feeds the load-balance auxiliary loss; routers that don't naturally produce one (e.g. `hierarchical`) fall back to a one-hot-of-topk approximation.

| Router          | Cost per layer (`d=2048, K=64`) | Mechanism |
|-----------------|---------------------------------|-----------|
| `linear`        | 131K, `O(d·K)`                  | `score = Linear(d, K)(x)` |
| `lowrank`       | 34K, `O(d·r_R + K·r_R)` at `r_R=16` | `score = (W_q x) · keys`, factored through a rank-`r_R` bottleneck |
| `cosine`        | 34K                              | `lowrank` with L2-normalized query and keys before the dot product |
| `hierarchical`  | 33K, `O(d·√K)`                   | Two-level: pick top-`√k` of `√K` groups, then top-`√k` experts within |
| `product_key`   | 33K, `O(d·√K)`                   | Lample et al. 2019: two scorers `d → √K`; expert `(i,j)` has additive score `s1[i] + s2[j]` |
| `multihead_pk`  | 131K at `H=4`                    | `H` parallel product-key heads, per-head K-wide scores averaged before top-k. Vectorized into one `Linear(d, H·√K)` per scorer. |
| `two_stage_pk`  | ~67K at `gate_rank=16`           | Stage 1: product-key picks top-k indices. Stage 2: a separate rank-`r_g` gate-calibration head recomputes soft-gate weights at the selected positions. `topk_weights = softmax(select_topk + gate_topk)` so both heads receive task-loss gradient. |
| `product_key_temp` | 33K + 1 scalar                | `product_key` + a learnable per-module temperature on the top-k softmax. Aux loss uses raw scores so the load-balance penalty does not pull tau back to 1. |
| `early_shared`  | ≈ 4K amortized                   | One `Linear(d, K)` at the first RoutedLoRA injection; the top-k decision is cached and reused at every later layer (Shleifer & Rush 2025). |

## Load-balance auxiliary loss

Switch-Transformer-style penalty (Fedus et al. 2021):

```
L_lb = K · Σ_i  f_i · p_i
f_i = fraction of tokens with expert i in their top-k
p_i = mean softmax probability of expert i across all tokens
```

`L_lb` is summed across all RoutedLoRA / DispatchMoELoRA modules in the model and added to the task loss with coefficient `aux_loss_coef = 0.01`. Mandatory at low `top_k` (`K=8, k=2` diverges without it under the linear router).

## Distillation auxiliary loss

When `--teacher_config` and `--teacher_ckpt` are provided to the training entry point, `RoutedLoRA.forward` additionally computes:

```
L_distill_module = KL(student_softmax(full_scores) || teacher_softmax(full_scores))
L_distill_total  = Σ_modules  L_distill_module
loss             = L_task + aux_loss_coef · L_lb + distill_coef · L_distill_total
```

The teacher forward is run with `torch.no_grad()` once per micro-batch; per-module `full_scores` are deposited on the matching student modules by qualified name. The teacher state is cleared after each backward pass so a stale dict cannot drive a future step.

The KL is on the K-wide score distribution at every RoutedLoRA module independently. With H = 4 module replicas at 32 injection points, that is 32 KL terms per micro-batch.

## Initialization

- LoRA `A`: `kaiming_uniform_(a=sqrt(5))` — standard.
- LoRA `B`: `zeros_` — keeps the residual zero at initialization, so the model behaves exactly like the frozen base at step 0.
- Router weights: standard `kaiming_uniform_` for `Linear(d, K)`-style modules; small `randn × 0.01` for explicit key parameters (`lowrank.keys`, `cosine.keys`, `two_stage_pk.gate_keys`).
- `product_key_temp.log_temperature`: `zeros_` so τ = 1 at init (identical to plain `product_key` at step 0).

## Numerical equivalence

`scripts/run_correctness.sh` (or `moe-lora-correctness`) verifies:

| Test | Comparison | Forward error | Gradient error |
|---|---|---|---|
| 1 | `MoELoRA` ≡ `DispatchMoELoRA` at matched `(K, r, top_k)` | `0.00e+00` | ≤ 1e-5 |
| 4 | `RoutedLoRA` ≡ `DispatchMoELoRA` at matched `(K, r, top_k)` | `0.00e+00` | ≤ 1e-5 |

By transitivity, all three implementations compute identical functions. The choice between them is purely about activation memory and compute cost.
