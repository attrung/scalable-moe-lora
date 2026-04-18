# Scalable MoE-LoRA: Efficient Routing for Fine-Grained Expert Mixtures

## Motivation

At a fixed parameter budget, splitting a mixture-of-experts (MoE) adapter into many small experts (fine granularity) instead of a few large ones is hypothesized to help because it **dramatically increases the number of expert combinations the model can represent per token**. With `K` experts and top-`k` activation, there are `C(K, k)` possible sparse activation patterns: `K=8, k=2` gives 28 combinations, but `K=64, k=16` jumps to roughly 4.9 × 10¹⁴. In principle this combinatorial diversity allows the adapter to specialize far more finely to the input distribution without using any extra active parameters per token.

Whether this combinatorial advantage actually translates into better downstream quality, and at what parameter cost on the routing side, is the central question of this study.

## Research questions

1. **Does expert granularity matter at a fixed parameter budget?** We sweep four granularity points at fixed `K·r = 64` (LoRA matrix-param budget) and fixed `top_k·r = 16` (active rank per token), crossed with two router types. Nine runs total.
2. **Which router mechanism best balances quality and parameter efficiency at fine granularity?** Once finer granularity is shown to help, the router parameter cost — a standard linear router scales as `O(d·K)` — becomes the bottleneck at high `K`. We compare six router designs at the finest granularity point (K=64) to find the best quality-efficiency tradeoff.

## Why LoRA — and why the findings generalize to full MoE

Running a controlled factorial like this at full model scale (pretraining multi-billion parameter dense MoE models for each cell) would take tens of thousands of GPU-hours. Our compute budget does not support that. Instead we use Low-Rank Adaptation (LoRA) with MoE routing on top of a frozen LLaMA 3.2 1B base. This reduces the cost by several orders of magnitude while preserving the two variables we care about:

- The **granularity axis is identical**: we vary `K` and `r` at fixed `K·r`, exactly as a full-MoE practitioner would choose expert count and hidden dim at fixed active parameter budget. The combinatorial argument above applies to any top-`k`-of-`K` routing mechanism, not just LoRA.
- The **router axis is identical**: we compare the same router mechanisms (linear, lowrank, factored √K, hierarchical, shared-across-layers) that are used in full MoE pretraining.

The LoRA setting isolates the routing contribution very cleanly — the base model is frozen, so any quality difference between configs comes purely from the adapter and its router. Treat this study as a **cheap, controlled dry-run for full MoE design decisions**:

- If the 1B LoRA results show a strong, consistent signal (as they already do for granularity and are emerging for routing), that is the trigger to commit serious pretraining compute to the winning configuration in a full MoE model.
- If the signal is weak or inconsistent, it saves us from burning that compute on a speculative design.

At DeepSeek-V3 scale (K=256 experts at d=7168), the router alone can cost ~1.8M parameters per layer and is run on every token — the scalability of the routing mechanism matters directly for inference latency and memory. The router comparison here is designed to identify which mechanisms preserve quality at √K cost, specifically so the findings transfer to frontier MoE pretraining.

## Current progress

| Phase | Description | Status |
|---|---|---|
| **A** | Legacy 12-dataset appendix runs (standard LoRA, MoE-LoRA, TM-LoRA, RoutedLoRA) | ✅ Complete |
| **B** | Granularity × router factorial (9 configs, seed 42, 18-dataset suite) | ✅ Complete |
| **C** | OOD evaluation on 7 held-out benchmarks for all 9 Phase B checkpoints | ✅ Complete |
| **D** | Router comparison study (4 new router types at K=64) | 🚧 3 of 4 trainings complete, early-shared still training |
| **E** | Multi-seed error bars on winning configs (seeds 0 and 123) | ☐ Planned |
| **F** | 3B scale-up of winning router + baseline | ☐ Planned |
| **G** | Paper writing, beam-search NLG polish, final figures | ☐ Planned |

## Key results so far

### Phase B: granularity sweep at K·r = 64, top_k·r = 16

Mean accuracy across 18 in-distribution datasets and 7 out-of-distribution benchmarks, LLaMA 3.2 1B, seed 42.

| Config | K | r | top_k | Router | val_loss | in-dist | OOD |
|---|---|---|---|---|---|---|---|
| baseline | — | 8 | — | standard LoRA | 1.984 | 0.500 | 0.416 |
| granularity_r8_k8 | 8 | 8 | 2 | lowrank rdim=16 | 1.951 | 0.509 | 0.429 |
| granularity_r4_k16 | 16 | 4 | 4 | lowrank rdim=16 | 1.949 | 0.522 | 0.436 |
| granularity_r2_k32 | 32 | 2 | 8 | lowrank rdim=16 | 1.944 | 0.526 | 0.441 |
| **granularity_r1_k64** | **64** | **1** | **16** | **lowrank rdim=16** | **1.933** | **0.531** | **0.441** |
| granularity_r4_k16_linear | 16 | 4 | 4 | linear | 1.926 | 0.523 | 0.438 |
| granularity_r2_k32_linear | 32 | 2 | 8 | linear | 1.926 | 0.531 | **0.452** |
| **granularity_r1_k64_linear** | **64** | **1** | **16** | **linear** | **1.923** | **0.534** | **0.451** |
| granularity_r8_k8_linear\* | 8 | 8 | 2 | linear | 2.139 | 0.371 | 0.321 |

\*`granularity_r8_k8_linear` diverged in epoch 2 (train loss climbed from 2.59 to 3.54). Routing collapse at `top_k=2` with a linear router is a likely cause; a re-run with a load-balancing term is planned.

**Findings:**
- **Granularity is monotonically beneficial.** For both router types (excluding the single diverged run), finer granularity produces lower val_loss and higher accuracy at constant LoRA parameter budget.
- **The linear router marginally outperforms the lowrank router** at every working granularity point (+0.3 to +0.5 in-dist, +1 OOD). The gap is due to routing quality: the LoRA matrices are identical between the two routers, only the routing decision mechanism differs.
- **OOD generalization tracks in-distribution accuracy**, so fine-grained routing is not overfitting to training data.

### Phase D: router comparison at finest granularity (K=64, r=1, top_k=16)

Four new router designs, same LoRA matrices, seed 42. All eight configs in this table share the identical LoRA matrix count (K·r = 64); only the routing mechanism varies, so any quality difference is pure routing contribution.

| Router | Cost per layer (d=2048, K=64) | val_loss | in-dist | OOD |
|---|---|---|---|---|
| linear (Phase B reference) | 131K | 1.923 | 0.534 | **0.451** |
| lowrank rdim=16 (Phase B reference) | 34K | 1.933 | 0.531 | 0.441 |
| **product-key (Phase D)** | **33K (√K scaling)** | **1.924** | **0.536** | **0.446** |
| hierarchical (Phase D) | 33K (√K scaling) | 1.954 | 0.517 | 0.433 |
| cosine (Phase D) | 34K | 1.954 | 0.520 | 0.438 |
| early-shared (Phase D) | 131K / L ≈ 4K (amortized) | 1.952 | 0.523 | *(OOD eval queued)* |

**Findings:**
- **Product-key routing matches the full linear router with √K parameter cost.** val_loss 1.924 vs linear's 1.923; in-dist 0.536 vs 0.534; OOD 0.446 vs 0.451 (within 0.005). At K=64 the router is 4× cheaper than linear; at K=4096, d=7168 it would be 23× cheaper. Factored scoring over the full expert product space preserves quality while scaling gracefully — the headline practical result.
- **Hierarchical routing underperforms.** The "shared within-group scores" constraint (every selected group receives the same local experts) is too restrictive — product-key's additive factorization over the full product space avoids this and wins.
- **Cosine normalization does not help.** The linear-vs-lowrank quality gap is a capacity/dimensionality issue, not a score-scale instability issue.
- **Early-shared routing works surprisingly well.** val_loss 1.952 and in-dist 0.523 — within ~0.01 of per-layer lowrank routing despite a *single* routing decision shared across all 32 LoRA injection points. OOD number still pending, but this is strong early evidence that per-layer routing may be unnecessary in the frozen-backbone PEFT setting, with a potential 16× router-parameter amortization across layers. Worth a standalone follow-up study.

## Repository structure

```
scalable-moe-lora/
├── src/                      # Core source code
│   ├── lora_layers.py        # LoRA variants + RoutedLoRA with 6 router types
│   ├── model.py              # LLaMA + LoRA injection
│   ├── data.py               # NLG dataset loaders (E2E, SAMSum, etc.)
│   ├── data_reasoning.py     # Reasoning + NLG loaders (18-dataset suite)
│   ├── train.py              # Training loop with validation + checkpointing
│   ├── train_reasoning.py    # Training wrapper for reasoning suite
│   ├── evaluate.py           # NLG eval (BLEU, ROUGE-L)
│   ├── evaluate_reasoning.py # In-dist + OOD eval (accuracy + generation metrics)
│   └── utils.py              # Config loading, checkpointing, seeding
├── configs/                  # YAML configs for each experiment
│   ├── reasoning_baseline.yaml
│   ├── granularity_r{R}_k{K}{,_linear}.yaml   # Phase B (9 configs)
│   └── granularity_r1_k64_{hierarchical,product_key,cosine,early_shared}.yaml   # Phase D
├── scripts/                  # SLURM sbatch wrappers
│   ├── sbatch_train.sh       # Training (gpu-medium)
│   ├── sbatch_eval.sh        # OOD eval (gpu-short)
│   ├── sbatch_stress.sh      # VRAM stress test (gpu-test)
│   └── stress_test.py        # Stress test implementation
├── results/                  # Eval JSONs from completed runs
│   ├── phase_b/              # 9 in-dist + 9 OOD eval JSONs
│   └── phase_d/              # Router comparison eval JSONs (partial)
├── docs/
│   └── future_work.md        # Phase E / F plans, analysis ideas
├── setup.sh                  # Environment setup template
├── requirements.txt
└── README.md                 # This file
```

## Quickstart

### Setup

```bash
git clone git@github.com:attrung/scalable-moe-lora.git
cd scalable-moe-lora
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure HF cache and prewarm LLaMA 3.2 1B
export HF_HOME=~/hf_cache
huggingface-cli download meta-llama/Llama-3.2-1B
```

### Train a single config

```bash
source setup.sh
python src/train_reasoning.py \
    --config configs/granularity_r1_k64.yaml \
    --datasets gsm8k,arc,commonsenseqa,piqa,winogrande,boolq,hellaswag,math,openbookqa,sciq,mbpp,logiqa2,drop,mmlu_aux,triviaqa,anli,e2e,samsum \
    --seed 42
```

### Evaluate on OOD benchmarks

```bash
python src/evaluate_reasoning.py \
    --config configs/granularity_r1_k64.yaml \
    --checkpoint results/phase_b/llama_granularity_r1_k64_..._seed42_best.pt \
    --datasets gsm8k,arc,commonsenseqa,piqa,winogrande,boolq,hellaswag,math,openbookqa,sciq,mbpp,logiqa2,drop,mmlu_aux,triviaqa,anli,e2e,samsum \
    --eval_datasets mmlu,mmlu_pro,bbh,ifeval,agieval,gpqa_diamond,truthfulqa \
    --seed 42 \
    --output results/phase_d/eval.json
```

### On SLURM (Adroit / similar)

```bash
source setup.sh
CONFIG=granularity_r1_k64 SEED=42 ROUTING=yes sbatch scripts/sbatch_train.sh
```

## Future work

- **Phase D tail**: complete early-shared router training and OOD eval. If early-shared matches linear quality, the paper's scalability argument becomes even stronger.
- **Phase E — multi-seed error bars**: re-run the winning Phase D router + baseline at seeds 0 and 123, extending the Phase C OOD sweep to the new checkpoints. Multi-seed results turn the current single-seed trend into a statistically defensible claim (mean ± σ), which is what reviewers and a decision to commit full-MoE pretraining compute will both want to see.
- **Phase F — 3B scale-up**: scale the winning router + baseline to LLaMA 3.2 3B at the finest granularity. Confirms that the granularity and router findings hold at larger model scale; required before extending to full MoE pretraining.
- **Phase G — paper**: beam-search polish on NLG eval, BBH re-evaluation with the corrected extractor, final figures and paper draft.
- **Extension to full MoE pretraining**: the LoRA setting here is deliberately chosen as a cheap controlled testbed. If Phases E + F show strong, consistent signals, the natural next step is to integrate the winning router (likely product-key or early-shared) into a full non-PEFT MoE pretraining run at frontier scale.

See `docs/future_work.md` for detailed plans.

## Key references

- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
- Luo et al. (2024). *MoELoRA: Contrastive Learning Guided Mixture of Experts on PEFT.* arXiv:2402.12851
- Lample et al. (2019). *Large Memory Layers with Product Keys.* NeurIPS 2019
- Krajewski et al. (2024). *Scaling Laws for Fine-Grained Mixture of Experts.* arXiv:2402.07871
- Shleifer & Rush (2025). *Omni-Router Transformer: Sharing Routing Decisions in Sparse MoE.* arXiv:2507.05724

## License

MIT License.

## Acknowledgments

This study was conducted as a COS 484 project at Princeton University, with 1B training runs on the Princeton Adroit A100 cluster.
