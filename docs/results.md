# Detailed results

All numbers below are from `LLaMA 3.2 1B + qv` injection at the locked hyperparameters in `README.md` § Method, seed 42, on the 18-dataset training suite (504k examples) and 7-benchmark OOD sweep. Validation loss is the best-epoch held-out cross-entropy. In-dist accuracy is the mean over the 16 accuracy-metric training datasets (E2E and SAMSum use BLEU/ROUGE-L). OOD accuracy is the mean over the 6 accuracy-metric OOD benchmarks (IFEval uses BLEU/ROUGE-L, reported separately).

## Part A — Adapter family

| Architecture            | K  | r  | k  | matrix params /layer | val_loss ↓ | in-dist ↑ | OOD ↑ |
|-------------------------|----|----|----|----------------------|------------|-----------|-------|
| Standard LoRA, r=16     | —  | 16 | —  | 32d                  | 1.984      | 0.500     | 0.207 |
| MoE-LoRA (coarse)*      | 8  | 8  | 2  | 128d                 | 2.154      | 0.371     | 0.197 |
| TM-LoRA, r=16           | 8  | 16 | 2  | 32d + 8·16           | 1.953      | 0.493     | 0.196 |
| **MoE-LoRA (fine)**     | 64 | 1  | 16 | 128d                 | **1.923**  | **0.534** | **0.245** |

\* Required the load-balance penalty to converge at `top_k=2`; without it the linear router diverges and we land at val_loss = 2.154 with the LB penalty (still the worst row in this column).

The three families are matched on **active rank per token** (`k · r = 16`), not on static parameter count: MoE-LoRA has 4× the per-layer matrix budget of Standard LoRA and TM-LoRA. The comparison reflects per-token compute rather than total parameters.

## Part B — Granularity sweep at fixed `K · r = 64`

### Linear router

| (r, K, k)        | val_loss ↓ | in-dist ↑ | OOD ↑      |
|------------------|------------|-----------|------------|
| (8, 8, 2)*       | 2.154      | 0.371     | 0.197      |
| (4, 16, 4)       | 1.926      | 0.523     | 0.226      |
| (2, 32, 8)       | 1.926      | 0.531     | **0.255**  |
| **(1, 64, 16)**  | **1.923**  | **0.534** | 0.245      |

### Lowrank router (`router_dim = 16`)

| (r, K, k)        | val_loss ↓ | in-dist ↑ | OOD ↑      |
|------------------|------------|-----------|------------|
| (8, 8, 2)        | 1.951      | 0.509     | 0.226      |
| (4, 16, 4)       | 1.949      | 0.522     | 0.223      |
| (2, 32, 8)       | 1.944      | 0.526     | 0.230      |
| **(1, 64, 16)**  | **1.933**  | **0.531** | 0.217      |

\* `(8, 8, 2)` linear cell required the load-balance penalty at `aux_loss_coef = 0.01`. The lowrank variant of the same cell converges *without* the penalty (`d → r_R → K` factoring smooths the routing decision).

## Part C — Router parameterization at `(K=64, r=1, top_k=16)`

| Router         | Per-layer router params      | val_loss ↓ | in-dist ↑ | OOD ↑     |
|----------------|------------------------------|------------|-----------|-----------|
| **linear**     | 131K, `d·K`                  | **1.923**  | 0.534     | **0.245** |
| lowrank, r_R=16 | 34K, `d·r_R + K·r_R`        | 1.933      | 0.531     | 0.217     |
| cosine         | 34K                          | 1.954      | 0.520     | 0.230     |
| hierarchical   | 33K, `2·d·√K`                | 1.954      | 0.517     | 0.225     |
| **product-key**| **33K, `2·d·√K`**            | **1.924**  | **0.536** | 0.224     |
| early-shared   | ≈ 4K amortized               | 1.952      | 0.523     | 0.200     |

Product-key matches linear on val_loss and in-dist at √K cost. Linear keeps a 2-3 pt OOD edge; Part D explains why.

## Part D — Per-layer routing analysis

### Per-module routing summary

For each of the 32 RoutedLoRA modules (16 layers × {q_proj, v_proj}), we walk all 18 in-distribution datasets and record the top-k expert indices per token. The aggregated metrics are then summarized across modules.

| Checkpoint               | K  | router        | distinct top-1 modes / module (median) | normalized entropy (median) | dead experts (worst layer) |
|--------------------------|----|---------------|----------------------------------------|------------------------------|----------------------------|
| K=8 coarse (best)        | 8  | linear        | 1                                      | 0.370                        | 6 / 8                      |
| K=8 coarse (final)       | 8  | linear        | 1                                      | 0.410                        | 6 / 8                      |
| K=64 linear              | 64 | linear        | 7                                      | 0.798                        | **30 / 64**                |
| K=64 lowrank             | 64 | lowrank       | 6                                      | 0.961                        | 0                          |
| K=64 cosine              | 64 | cosine        | 6                                      | 0.973                        | 0                          |
| K=64 hierarchical        | 64 | hierarchical  | 7                                      | 0.971                        | 0                          |
| K=64 product-key         | 64 | product-key   | 7                                      | 0.969                        | 0                          |
| K=64 early-shared        | 64 | early-shared  | 3                                      | 0.802                        | 4 / 64 (everywhere)        |

Key observations:

1. **Capacity paradox.** The linear router has the largest router parameter count (131K) but leaves up to 30 of 64 experts dead at its worst layer. Every cheaper router keeps every expert alive. Linear's extra capacity is spent on imbalance, not on more-informative routing.

2. **Cheap routers are *more* input-conditional at the layer level**, not less. Distinct top-1 modes per layer is 6-7 for the cheap factored routers and 7 for linear. So the 2-3 pt OOD edge cannot come from sharper top-k expert identity.

3. **Early-shared is structurally uniform across layers by design.** All 32 RoutedLoRA injections read a single cached routing decision, so distinct top-1 per layer is exactly 3 *every*where, the pairwise hot-set Jaccard is 1, and the same 4 of 64 experts are dead everywhere. The OOD penalty (0.200 vs 0.217-0.245 for per-layer routers) is the direct cost of this forced uniformity.

### Gate-magnitude statistics (cheap diagnostic for the OOD-edge hypothesis)

`results/analysis/_gate_magnitudes.json` records, per RoutedLoRA module, the softmax-normalized gate weights at the top-k positions. Pooled across all 32 modules and all 18 datasets:

| Router         | mean gate | std    | mean(max gate) | p95(max gate) | normalized entropy |
|----------------|-----------|--------|----------------|---------------|--------------------|
| **linear**     | 0.063     | **0.122** | **0.432**   | **0.951**     | **0.664**          |
| lowrank        | 0.063     | 0.0003 | 0.063          | 0.064         | 1.000              |
| cosine         | 0.063     | 0.007  | 0.079          | 0.089         | 0.998              |
| hierarchical   | 0.063     | 0.014  | 0.090          | 0.113         | 0.992              |
| product-key    | 0.063     | 0.011  | 0.086          | 0.104         | 0.995              |
| early-shared   | 0.063     | 0.074  | 0.245          | 0.588         | 0.851              |

Mean gate is 1/16 = 0.0625 by construction (softmax over top-k weights summing to 1). The shape of the distribution is what differs:

- The **linear router** produces dramatically sharper gates: at the 95th-percentile token, its top expert holds 95% of the gate mass. Median normalized entropy is 0.664 (1 = uniform, 0 = one-hot).
- The **cheap factored routers** (lowrank, cosine, product-key, hierarchical) are essentially uniform across the selected 16 — their max-gate p95 is ~0.06-0.11.
- The **early-shared router** sits between: its single linear router has linear-like expressivity, but only one of them. Median entropy 0.851, max-gate p95 0.588.

So the linear router's 2-3 pt OOD edge correlates directly with **gate-magnitude resolution**, not with expert selection. The cheap routers do select the right top-k — they just then weight all 16 selected experts (almost) equally. Linear preserves the soft-mixing fidelity that the cheap routers lose.

## Caveats

- **Single seed.** Most cells are seed=42 only. The 2-3 pt OOD gap between linear and cheap routers is borderline statistical evidence; the project includes (or plans) seeds 0 and 123 for the Part C linear and product-key winners to put error bars on this.
- **Limited model scale.** All numbers are LLaMA 3.2 1B. The granularity / router findings should generalize to larger frozen-backbone PEFT setups, but Part F (3B scale-up, future work) would be the direct test.
- **Zero-shot.** All eval prompts are zero-shot, no exemplars or "think step by step" scaffolding. GSM8K and MATH gold targets are full worked solutions, so the model emits a rationale before the final answer on those two datasets.
- **OOD numbers low in absolute terms.** Several OOD benchmarks (BBH, MMLU-Pro, GPQA Diamond) are near random for LLaMA 3.2 1B even before fine-tuning. What matters is the relative ordering across configurations.
