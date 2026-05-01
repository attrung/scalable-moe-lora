# Scalable MoE-LoRA

Controlled factorial study of mixture-of-experts (MoE) routing in low-rank adapters on a frozen LLaMA 3.2 1B backbone. Three connected research questions plus a post-hoc per-layer routing analysis, evaluated on a 18-dataset training suite (≈504k examples) and a 7-benchmark out-of-distribution sweep.

```
Standard LoRA      MoE-LoRA            TM-LoRA
    │                 │                   │
    │            ╔════╧═════════╗         │
    │            ║ Part A: WHAT ║         │
    │            ║   ARCH WINS? ║         │
    │            ╚══════╤═══════╝         │
    │                   │                  │
                fine‑grained MoE‑LoRA wins
                        │
        ╔═══════════════╧═══════════════╗
        ║ Part B: HOW SHOULD WE SPLIT   ║
        ║   THE EXPERT BUDGET?          ║
        ║   K · r = 64,  k · r = 16     ║
        ╚═══════════════╤═══════════════╝
                        │
        finer is monotonically better in‑dist;
        OOD peaks one step coarser
                        │
        ╔═══════════════╧═══════════════╗
        ║ Part C: WHICH ROUTER GIVES    ║
        ║   THE BEST QUALITY/COST?      ║
        ╚═══════════════╤═══════════════╝
                        │
        product‑key matches linear at √K cost
        in‑dist; linear keeps a 2‑3 pt OOD edge
                        │
        ╔═══════════════╧═══════════════╗
        ║ Part D: PER-LAYER ROUTING     ║
        ║   ANALYSIS — WHY?             ║
        ╚═══════════════╤═══════════════╝
                        │
        linear router has dramatically sharper
        soft gates (max gate p95 = 0.95 vs 0.06–0.11
        for cheap routers); the OOD edge is
        gate‑magnitude fidelity, not selection
        sharpness
```

## TL;DR

We hold the parameter budget fixed (`K·r = 64` matrix params per layer, `k·r = 16` active rank per token) and sweep the design axes that matter for an MoE-LoRA adapter:

| Axis | Cells | Headline finding |
|---|---|---|
| **A. Architecture** | Standard LoRA, MoE-LoRA (coarse + fine), TM-LoRA | Fine-grained MoE-LoRA wins in-dist; standard LoRA has a small OOD edge over the table-style TM-LoRA. |
| **B. Granularity** | `(r, K, k) ∈ {(8,8,2), (4,16,4), (2,32,8), (1,64,16)}` × {linear, lowrank} routers | Finer granularity is monotonically better in-distribution; OOD peaks one step coarser at `r=2, K=32`. |
| **C. Router** | linear, lowrank, cosine, hierarchical, product-key, early-shared at `K=64` | Product-key matches the full linear router on val_loss and in-dist at √K parameter cost. Linear keeps a 2-3 pt OOD edge. |
| **D. Per-layer routing** | trained-checkpoint inspection of all 8 routers | Linear's OOD edge is **soft-gate magnitude fidelity** (max gate p95 = 0.95 vs ≈ 0.06-0.11 for the cheap routers), not sharper top-k selection. |

The full linear router at `K=64` even leaves up to **30 of 64 experts dead** at its worst layer — the cheap factored routers keep all 64 alive. The linear router's extra capacity is spent on **imbalance**, not on more-informative routing.

## Why a LoRA testbed for MoE design

Running this factorial at full pretraining scale would burn tens of thousands of GPU-hours per cell. A frozen LLaMA 3.2 1B + low-rank adapter setup compresses the cost by orders of magnitude while preserving the two variables we care about:

1. **Granularity is identical.** Varying `K` and `r` at fixed `K·r` is the same choice a full-MoE practitioner makes between many small experts and a few large ones.
2. **Router parameterization is identical.** The router mechanisms compared here (linear, lowrank, cosine, hierarchical, product-key, early-shared) are the same family used in production MoE pretraining.

At DeepSeek-V3 scale (`K=256, d=7168`) the router alone costs ~1.8M parameters per layer and runs on every token. The √K-cost product-key router observed here would extrapolate to ~23× cheaper with no loss of in-distribution quality — a directly relevant signal before committing pretraining compute to a routing design.

## Repository layout

```
scalable-moe-lora/
├── src/scalable_moe_lora/        # importable Python package
│   ├── adapters/                 # adapter zoo + router catalogue
│   │   ├── base.py               #   LoRA, LinearWithLoRA wrapper
│   │   ├── moe.py                #   MoELoRA (K experts, top-k routed)
│   │   ├── tm.py                 #   TM-LoRA
│   │   └── routers.py            #   9 router types + build_router()
│   ├── data/                     # dataset loaders
│   │   ├── nlg.py                #   NLG primitives (E2E, SAMSum, ...)
│   │   └── reasoning.py          #   18-dataset training suite + 7 OOD
│   ├── analysis/                 # routing-behavior analysis
│   │   ├── per_layer_routing.py  #   per-module top-k indices across datasets
│   │   ├── gate_magnitudes.py    #   gate sharpness statistics
│   │   └── per_layer_summary.py  #   entropy / dead-expert / Jaccard summaries
│   ├── model.py                  # LLaMA + adapter injection
│   ├── train.py                  # training loop (resumable, distillation-aware)
│   ├── train_reasoning.py        # reasoning-suite training entry point
│   ├── evaluate.py               # NLG eval (BLEU, ROUGE-L)
│   ├── evaluate_reasoning.py     # in-dist + OOD eval + routing capture
│   └── utils.py
├── configs/                      # YAML experiment configs
│   ├── a_architecture/           #   4 cells (Part A)
│   ├── b_granularity/            #   8 cells (Part B): linear/, lowrank/
│   ├── c_routers/                #   6 cells (Part C): linear, lowrank, cosine,
│   │                             #   hierarchical, product_key, early_shared
│   └── extensions/               #   multihead-PK, two-stage-PK, temperature,
│                                 #   linear-→-PK distillation
├── scripts/                      # local launchers + Slurm wrappers
│   ├── run_train.sh              #   single-config local training
│   ├── run_eval.sh               #   single-checkpoint local eval
│   ├── run_distill.sh            #   teacher → student distillation
│   ├── migrate_multihead_checkpoint.py
│   └── slurm/                    #   Slurm sbatch templates
├── results/                      # eval JSONs + analysis JSONs (committed)
│   ├── eval/                     #   per-checkpoint in-dist + OOD evaluation
│   └── analysis/                 #   per-layer routing + gate-magnitude data
├── docs/
│   ├── reproduce.md              #   step-by-step reproduction
│   ├── architecture.md           #   adapter and router math
│   └── results.md                #   detailed result tables
├── tests/
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

## Quickstart

### Install

```bash
git clone https://github.com/attrung/scalable-moe-lora.git
cd scalable-moe-lora

python3 -m venv venv
source venv/bin/activate

pip install -e .          # installs scalable-moe-lora and its dependencies
huggingface-cli login     # required to download LLaMA 3.2 1B
huggingface-cli download meta-llama/Llama-3.2-1B
```

The package exposes four console scripts after install:

| Command | What it does |
|---|---|
| `moe-lora-train`         | Train a single configuration from a YAML config |
| `moe-lora-eval`          | In-dist + (optional) OOD evaluation on a checkpoint |
| `moe-lora-routing`       | Per-layer top-k routing collection (CPU) |
| `moe-lora-gates`         | Per-layer gate-magnitude statistics (CPU) |

### Train a single cell

```bash
# Local (single GPU, ~16 h on a single A100 80 GB at the default config):
./scripts/run_train.sh configs/c_routers/linear.yaml 42

# Slurm:
CONFIG=configs/c_routers/linear.yaml SEED=42 \
    sbatch scripts/slurm/train.sbatch
```

### Evaluate a saved checkpoint

```bash
# In-distribution only (~0.5 h):
./scripts/run_eval.sh \
    configs/c_routers/linear.yaml \
    results/checkpoints/routers_linear_..._best.pt

# In-distribution + OOD (~1.5 h):
./scripts/run_eval.sh \
    configs/c_routers/linear.yaml \
    results/checkpoints/routers_linear_..._best.pt 42 \
    --eval_datasets mmlu,mmlu_pro,bbh,ifeval,agieval,gpqa_diamond,truthfulqa
```

### Per-layer routing analysis

```bash
# Copy the manifest template and edit checkpoint paths:
cp configs/analysis_manifest.example.yaml results/analysis_manifest.yaml
$EDITOR results/analysis_manifest.yaml

# Collect (CPU-only, ~30 min for 6 K=64 checkpoints at 5 samples/dataset):
moe-lora-routing --manifest results/analysis_manifest.yaml
moe-lora-gates   --manifest results/analysis_manifest.yaml
python -m scalable_moe_lora.analysis.per_layer_summary
```

## Method

**Base model.** Frozen LLaMA 3.2 1B (16 decoder layers, hidden dim `d=2048`). Adapters attached to query and value projections at every layer (32 injection points total).

**Training data.** 18-dataset suite (~504k examples) covering arithmetic reasoning (GSM8K, MATH), multiple-choice science and commonsense (ARC, CommonsenseQA, PIQA, WinoGrande, BoolQ, HellaSwag, OpenBookQA, SciQ, MMLU-aux, LogiQA2), reading comprehension and QA (DROP, TriviaQA, ANLI), code (MBPP), and generation (E2E, SAMSum).

**OOD benchmarks.** 7 held-out datasets: MMLU, MMLU-Pro, BBH, IFEval, AGIEval, GPQA Diamond, TruthfulQA.

**Hyperparameters** (locked across the factorial, `configs/*.yaml`):

| | |
|---|---|
| Effective batch | 36 (micro-batch 12 × grad-accum 3) |
| Sequence length | 1024 |
| Learning rate    | 4e-4 (3e-4 for TM-LoRA), linear warmup 500 steps then linear decay |
| Weight decay    | 0.01 |
| LoRA dropout    | 0.1 |
| Label smoothing | 0.1 |
| Epochs          | 3 |
| Aux loss coef   | 0.01 (Switch-Transformer load balance, mandatory at low `top_k`) |
| Precision       | base fp16 frozen, adapter fp32 |
| Gradient ckpt   | on (`use_reentrant=False`) |
| Primary seed    | 42 |

**Evaluation.** Zero-shot prompts (no in-context exemplars or "think step by step" scaffolding). Gold targets are short answers except on GSM8K and MATH (full worked solutions, model emits a rationale before the final answer). In-distribution accuracy is the mean over the **15** accuracy-metric training datasets (MBPP, E2E, and SAMSum use BLEU/ROUGE-L and are excluded from the in-dist average). OOD accuracy is the mean over the 6 accuracy-metric OOD benchmarks. **IFEval is excluded from the OOD pool**: its "gold reference" is the prompt's first sentence (`data/reasoning.py:format_ifeval`), so a BLEU/ROUGE-L score there measures prompt-echoing, not instruction-following. Validation loss is the best-epoch held-out cross-entropy on the training suite's validation split.

## Results

### Part A — Architecture family

At matched active rank per token (`k · r = 16`), at `LLaMA 3.2 1B + qv` injection:

| Architecture            | K  | r  | k  | val_loss ↓ | in-dist ↑ | OOD ↑ |
|-------------------------|----|----|----|------------|-----------|-------|
| Standard LoRA, r=16     | —  | 16 | —  | 1.984      | 0.5111    | 0.227 |
| MoE-LoRA (coarse)       | 8  | 8  | 2  | 2.154      | 0.371*    | 0.195 |
| TM-LoRA, r=16           | 8  | 16 | 2  | 1.953      | 0.5088    | 0.220 |
| **MoE-LoRA (fine)**     | 64 | 1  | 16 | **1.923**  | **0.534** | **0.245** |

\* coarse cell at `k=2` requires the load-balance penalty to converge under the linear router; without it the cell diverges. Two seed=42 runs of this cell (`routed_r8_k8_linear` and `llama_granularity_r8_k8_linear`) had divergent training trajectories from unseeded RNG paths in the router init; the row above uses the more stable run, picked by best validation loss. After the `PYTHONHASHSEED` fix on this branch, future re-runs of this cell will be deterministic. The MoE-LoRA (fine) row is the same checkpoint as Part B's `r=1, K=64, linear` cell, reused here so both ends of the granularity dimension show inside Part A.

### Part B — Granularity sweep

Held: `K · r = 64` (matrix budget) and `k · r = 16` (active rank per token). Sweeped: 4 splits × 2 routers (linear, lowrank with r_R=16).

| (r, K, k) | linear router |       |       | lowrank router |       |       |
|-----------|---------------|-------|-------|----------------|-------|-------|
|           | val_loss      | in    | OOD   | val_loss       | in    | OOD   |
| (8, 8, 2)*| 2.154         | 0.371 | 0.195 | 1.951          | 0.509 | 0.226 |
| (4, 16, 4)| 1.926         | 0.523 | 0.226 | 1.949          | 0.522 | 0.223 |
| (2, 32, 8)| 1.926         | 0.531 | **0.255** | 1.944      | 0.526 | 0.230 |
| **(1, 64, 16)** | **1.923** | **0.534** | 0.245 | **1.933**  | **0.531** | 0.217 |

\*linear at `k=2, K=8` requires the load-balance penalty.

Three findings:
1. **Finer granularity is monotonically better in-distribution** under both routers.
2. **OOD peaks one step coarser** at `(r=2, K=32)` — a small generalization penalty at the finest split.
3. **The lowrank router is structurally more robust at low top-k**: it converges at `(K=8, r=8, k=2)` *without* the load-balance penalty; the linear router at the same cell diverges. The `d → r_R → K` factoring smooths the routing decision enough to avoid winner-take-all collapse.

### Part C — Router parameterization

Held: `(K=64, r=1, top_k=16)`. Six router parameterizations:

| Router          | per-layer params | val_loss ↓ | in-dist ↑ | OOD ↑     |
|-----------------|------------------|------------|-----------|-----------|
| **linear**      | 131K             | **1.923**  | 0.534     | **0.245** |
| lowrank, r_R=16 | 34K              | 1.933      | 0.531     | 0.217     |
| cosine          | 34K              | 1.954      | 0.520     | 0.230     |
| hierarchical    | 33K (√K cost)    | 1.954      | 0.517     | 0.225     |
| **product-key** | **33K (√K cost)**| **1.924**  | **0.536** | 0.224     |
| early-shared    | ≈ 4K amortized   | 1.952      | 0.523     | 0.200     |

Headline: **product-key matches linear on val_loss and in-distribution accuracy at √K parameter cost** (a 4× router-parameter reduction at this scale; ~23× at frontier-MoE scale `K=4096, d=7168`). The full linear router keeps a 2-3 pt OOD advantage over every √K-cost alternative, which Part D explains via soft-gate fidelity rather than selection sharpness.

### Part D — Per-layer routing analysis

We open the trained checkpoints of every Part C router and inspect routing behavior per MoELoRA module across all 18 in-dist datasets.

**Capacity paradox.** Despite having the largest per-layer router parameter count (131K), the linear router leaves **30 of 64 experts dead** in its worst layer (no other K=64 router has any dead experts in any layer). Median per-layer routing entropy is 0.798 for linear vs 0.960-0.973 for the factored routers. The linear router's extra capacity is spent on imbalance, not on more-informative routing.

**Gate-magnitude fidelity.** On the K-wide softmax over selected experts, the linear router produces dramatically sharper gates than the cheap routers:

| Router        | mean  | std    | mean(max) | p95(max) | normalized entropy |
|---------------|-------|--------|-----------|----------|--------------------|
| **linear**    | 0.063 | **0.122** | **0.432** | **0.951** | **0.664**       |
| lowrank       | 0.063 | 0.0003 | 0.063     | 0.064    | 1.000              |
| cosine        | 0.063 | 0.007  | 0.079     | 0.089    | 0.998              |
| hierarchical  | 0.063 | 0.014  | 0.090     | 0.113    | 0.992              |
| product-key   | 0.063 | 0.011  | 0.086     | 0.104    | 0.995              |
| early-shared  | 0.063 | 0.074  | 0.245     | 0.588    | 0.819              |

Linear's max gate at the 95th-percentile token holds 95% of the weight; the cheap factored routers are essentially uniform 1/16 = 0.0625. This is direct evidence that the linear router's 2-3 pt OOD edge comes from **soft-gate magnitude resolution**, not from selecting different experts. The cheap routers do select the right top-k — they just then weight all 16 selected experts (almost) equally.

**Distinct top-1 selections.** The cheap factored routers (lowrank, cosine, product-key, hierarchical) post 6-7 distinct dataset-level top-1 modes per layer; the linear router sits at 5. By that metric, the cheap routers are *more* input-conditional at the layer level, ruling out "linear has a sharper top-k" as the OOD-edge explanation. (Hierarchical's level-2 router is shared across selected groups by design — see `docs/architecture.md` § Router catalogue — so its high distinct-top-1 score reflects diversity in the level-1 group decision rather than independent expert decisions per group.)

## Extensions

Four follow-up cells aimed at the §Part D gate-fidelity hypothesis (still in flight as of this writeup):

| Extension          | Tests                                                              | Status |
|--------------------|--------------------------------------------------------------------|--------|
| `multihead_pk`     | OOD edge = "rank of the gate function"? (H=4 PK heads, linear-cost params) | running |
| `two_stage_pk`     | OOD edge = dedicated gate-calibration head?                       | running |
| `product_key_temp` | Cheapest possible: a learnable per-module temperature on the top-k softmax | queued |
| `distill_pk_from_linear` | Direct transfer of the teacher's K-wide gate distribution via KL | queued |

When complete, each extension's eval JSONs land in `results/eval/` and feed into Part C's row of `docs/results.md`.

## Reproduction

A single-seed pass through Parts A-C is roughly 20 cells × ~16 h = ~320 GPU-hours on an A100 80 GB. See `docs/reproduce.md` for a step-by-step.

## Citation

```bibtex
@misc{nguyen2026scalablemoelora,
  author = {Nguyen, Anh Trung and Lu, Jiaqi and Huang, Andrew},
  title  = {Scalable MoE-LoRA: A Controlled Factorial Study of Mixture-of-Experts Routing for Low-Rank Adaptation},
  year   = {2026},
  url    = {https://github.com/attrung/scalable-moe-lora},
}
```

## License

[MIT](LICENSE).

## References

- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
- Luo et al. (2024). *MoELoRA: Contrastive Learning Guided Mixture of Experts on PEFT.* arXiv:2402.12851
- Lample et al. (2019). *Large Memory Layers with Product Keys.* NeurIPS 2019
- Krajewski et al. (2024). *Scaling Laws for Fine-Grained Mixture of Experts.* arXiv:2402.07871
- Shleifer & Rush (2025). *Omni-Router Transformer: Sharing Routing Decisions in Sparse MoE.* arXiv:2507.05724
- Fedus et al. (2021). *Switch Transformers: Scaling to Trillion Parameter Models.* arXiv:2101.03961

## Acknowledgments

This study was conducted as a final project for COS 484 (Princeton) on the Princeton Adroit cluster. Training was performed on 1× NVIDIA A100 80 GB per cell.
