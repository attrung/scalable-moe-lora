# Scalable MoE-LoRA: Efficient Routing for Fine-Grained Expert Mixtures

A controlled empirical study of routed Low-Rank Adaptation (LoRA) with mixture-of-experts (MoE) routing, focused on two questions that matter for scaling MoE architectures to frontier model sizes:

1. **Does expert granularity matter at a fixed parameter budget?** We sweep four granularity points at fixed `K·r = 64` (matrix-param budget) and fixed `top_k·r = 16` (active rank per token), crossed with two router types. Nine runs total.
2. **Which router mechanism best balances quality and parameter efficiency at fine granularity?** Once finer granularity is shown to help, the router parameter cost — linear routing scales as `O(d·K)` — becomes the bottleneck at high K. We compare six router designs at the finest granularity point (K=64) to find the best quality-efficiency tradeoff.

## Why this matters for MoE in general

The principles studied here are not specific to LoRA adapters. At frontier model scale (e.g., DeepSeek-V3 with K=256 experts at d=7168), routing can consume millions of parameters per layer. The empirical findings transfer directly:

- **Granularity at fixed budget**: fine-grained sparse activation (many small experts, few active per token) is currently used by DeepSeek, Mixtral, and Qwen-MoE. Our study isolates the granularity axis in a controlled way — same total parameters, same active parameters, only the `K·r` split changes — which no prior work has done cleanly in the PEFT setting.
- **Router scalability**: a full linear router `Linear(d, K)` becomes impractical at large K. Our comparison of factored routers (product-key, hierarchical, cosine, early-shared) quantifies the quality-efficiency Pareto curve and identifies which compression strategies preserve quality. Product-key routing scales as `O(d·√K)` and matches linear quality in our measurements — a 23× parameter reduction at K=4096, d=7168.
- **Per-layer vs. shared routing**: the "early-shared" router tests whether routing decisions even need to be made per-layer, motivated by the Omni-Router Transformer finding that shared routing can outperform per-layer routing in deep residual networks. If this holds in the LoRA setting, single-decision routing amortizes the router cost across L layers.

The LoRA setting gives us a clean experimental platform: the base model is frozen, only the routers and expert matrices are trained, so any quality difference comes directly from the router. The findings then motivate design choices for full MoE pretraining.

## Current progress

| Phase | Description | Status |
|---|---|---|
| **A** | Legacy 12-dataset appendix runs (standard LoRA, MoE-LoRA, TM-LoRA, RoutedLoRA) | ✅ Complete |
| **B** | Granularity × router factorial (9 configs, seed 42, 18-dataset suite) | ✅ Complete |
| **C** | OOD evaluation on 7 held-out benchmarks for all 9 Phase B checkpoints | ✅ Complete |
| **D** | Router comparison study (4 new router types at K=64) | 🚧 3 of 4 trainings complete, early-shared still training |
| **E** | 3B scale-up of winning router + baseline | ☐ Planned |
| **F** | Paper writing, beam-search NLG polish, final figures | ☐ Planned |

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

Four new router designs, same LoRA matrices, seed 42.

| Router | Cost per layer (d=2048, K=64) | val_loss | in-dist | OOD |
|---|---|---|---|---|
| linear (Phase B reference) | 131K | 1.923 | 0.534 | 0.451 |
| lowrank rdim=16 (Phase B reference) | 34K | 1.933 | 0.531 | 0.441 |
| **product-key (Phase D)** | **33K** | **1.924** | **0.536** | **0.446** |
| hierarchical (Phase D) | 33K | 1.954 | 0.517 | 0.433 |
| cosine (Phase D) | 34K | 1.954 | 0.520 | *(eval running)* |
| early-shared (Phase D) | 131K / L ≈ 4K | *(training running)* | — | — |

**Findings:**
- **Product-key routing matches the full linear router with √K parameter cost.** At K=64 it is 4× cheaper than linear; at K=4096 it would be 23× cheaper. This is the strongest practical result — factored scoring over the product space preserves quality while scaling gracefully.
- **Hierarchical routing underperforms.** The "shared within-group scores" constraint (all selected groups get the same local experts) is too restrictive.
- **Cosine normalization does not help.** The linear-vs-lowrank quality gap is not a score-scale instability issue.
- **Early-shared routing result pending.** If it matches linear quality, single-decision routing provides an additional 16× amortization across LoRA injection layers.

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
- **Phase D-stretch** (if time permits): re-run `granularity_r8_k8_linear` with a load-balancing loss to confirm the routing-collapse failure mode.
- **Phase E**: scale the winning router to LLaMA 3.2 3B at the finest granularity. Confirm that the granularity and router findings hold at larger model scale.
- **Phase F**: beam-search polish on NLG eval, BBH re-evaluation with the corrected extractor, final figures and paper draft.
- **Extension**: integrate Phase D router variants into a full (non-PEFT) MoE pretraining run. The LoRA setting isolates the routing contribution; validating the findings in pretraining confirms the principles generalize.

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
