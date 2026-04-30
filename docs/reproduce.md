# Reproduction guide

Step-by-step walk-through of a full reproduction of Parts A-D plus extensions.

## Compute budget

| | per-cell | total |
|---|---|---|
| Part A      | ~16 h × 4 cells  | ~64 h GPU |
| Part B      | ~16 h × 8 cells  | ~128 h GPU |
| Part C      | ~16 h × 6 cells  | ~96 h GPU |
| Part D analysis | ~30 min × 1 (CPU) | negligible |
| Extensions  | ~16-30 h × 4 cells | ~80 h GPU |

Total ~370 GPU-hours on a single A100 80 GB. Multi-seed (seeds 0, 123) on the Part C winners adds another ~32 h GPU per seed.

## 1. Environment

```bash
git clone https://github.com/<org>/scalable-moe-lora.git
cd scalable-moe-lora
python3 -m venv venv && source venv/bin/activate
pip install -e .

huggingface-cli login
huggingface-cli download meta-llama/Llama-3.2-1B
export HF_HOME=$(pwd)/hf_cache
```

If running on a Slurm cluster, set `REPO_ROOT` and (optionally) `ACTIVATE` env vars before submitting:

```bash
export REPO_ROOT=/path/to/scalable-moe-lora
export ACTIVATE=$REPO_ROOT/venv/bin/activate
```

## 2. Sanity checks (~5 min)

```bash
# Numerical equivalence of the 3 MoE-LoRA implementations
moe-lora-correctness

# 50-step smoke run with validation
./scripts/run_train.sh configs/c_routers/linear.yaml 42 \
    --max_steps 50 --smoke_with_val
```

The correctness test should print forward error `0.00e+00` and gradient error `≤ 1e-5` for both Test 1 (`MoELoRA ≡ DispatchMoELoRA`) and Test 4 (`RoutedLoRA ≡ DispatchMoELoRA`).

## 3. Part A — Architecture (4 cells)

Train each cell separately:

```bash
for cfg in configs/a_architecture/*.yaml; do
    CONFIG=$cfg SEED=42 sbatch scripts/slurm/train.sbatch
done
```

After training, evaluate each checkpoint on in-dist + OOD:

```bash
for cfg in configs/a_architecture/*.yaml; do
    name=$(basename $cfg .yaml)
    CKPT=results/checkpoints/${name}_<DS_TAG>_seed42_best.pt
    CONFIG=$cfg CKPT=$CKPT OOD=yes sbatch scripts/slurm/eval.sbatch
done
```

Replace `<DS_TAG>` with `gsm8k+arc+commonsenseqa+piqa+winogrande+boolq+hellaswag+math+openbookqa+sciq+mbpp+logiqa2+drop+mmlu_aux+triviaqa+anli+e2e+samsum`.

## 4. Part B — Granularity (8 cells)

```bash
for router in linear lowrank; do
    for cell in r8_k8 r4_k16 r2_k32 r1_k64; do
        CONFIG=configs/b_granularity/$router/$cell.yaml SEED=42 \
            sbatch scripts/slurm/train.sbatch
    done
done
```

The `linear/r8_k8` cell requires the load-balance penalty (already set via `aux_loss_coef: 0.01` in the YAML). Without it the cell diverges in epoch 2.

Evaluate (in-dist + OOD) per the same pattern as Part A.

## 5. Part C — Router parameterization (6 cells)

```bash
for router in linear lowrank cosine hierarchical product_key early_shared; do
    CONFIG=configs/c_routers/$router.yaml SEED=42 \
        sbatch scripts/slurm/train.sbatch
done
```

The `early_shared` router has a non-trivial owner-of-the-cache state set up in `model.py:_set_early_shared_owner` — only the first RoutedLoRA injection holds the router parameter, downstream injections read from its cache. The total per-layer cost is therefore ~131K / L ≈ 4K amortized.

## 6. Part D — Per-layer routing analysis (CPU, ~30 min)

Once Part C checkpoints are saved:

```bash
cp configs/analysis_manifest.example.yaml results/analysis_manifest.yaml
$EDITOR results/analysis_manifest.yaml   # fill in the actual checkpoint paths

moe-lora-routing  --manifest results/analysis_manifest.yaml
moe-lora-gates    --manifest results/analysis_manifest.yaml
python -m scalable_moe_lora.analysis.per_layer_summary
```

Outputs:
- `results/analysis/<tag>.json` — per-module routing data per dataset
- `results/analysis/_gate_magnitudes.json` — gate sharpness statistics
- `results/analysis/_summary.json` — cross-checkpoint summary table

The summary script also prints a human-readable Markdown table to stdout for direct inclusion in the writeup.

## 7. Extensions (optional)

```bash
# Multi-head product-key (linear-cost params, factored structure)
CONFIG=configs/extensions/multihead_pk.yaml SEED=42 \
    sbatch scripts/slurm/train.sbatch

# Two-stage product-key (PK selection + rank-r_g gate calibration)
CONFIG=configs/extensions/two_stage_pk.yaml SEED=42 \
    sbatch scripts/slurm/train.sbatch

# Per-module learnable temperature on top-k softmax
CONFIG=configs/extensions/product_key_temp.yaml SEED=42 \
    sbatch scripts/slurm/train.sbatch

# Distillation: product-key student from linear-router teacher
STUDENT_CONFIG=configs/extensions/distill_pk_from_linear.yaml \
    TEACHER_CONFIG=configs/c_routers/linear.yaml \
    TEACHER_CKPT=results/checkpoints/routers_linear_<DS_TAG>_seed42_best.pt \
    DISTILL_COEF=1.0 SEED=42 \
    sbatch scripts/slurm/distill.sbatch
```

## 8. Multi-seed error bars (optional, +6 cells)

For the Part C winner (and reference linear router):

```bash
for seed in 0 123; do
    for router in linear product_key; do
        CONFIG=configs/c_routers/$router.yaml SEED=$seed \
            sbatch scripts/slurm/train.sbatch
    done
done
```

## Resuming from a wall-time-out

If a Slurm job hits its wall before completing all 3 epochs:

```bash
CONFIG=configs/extensions/multihead_pk.yaml SEED=42 \
    RESUME_FROM=results/checkpoints/ext_multihead_pk_<DS_TAG>_seed42_best.pt \
    sbatch scripts/slurm/train.sbatch
```

`--resume_from` restores model + optimizer state and advances the LR scheduler to the saved step. The outer epoch loop starts at the next epoch index, so a checkpoint saved at the end of epoch 2 resumes by training only epoch 3.

## Layout migration for old multihead-PK checkpoints

The current `MultiHeadProductKeyRouter` uses one `Linear(d, H·√K)` per scorer instead of `H` separate `Linear(d, √K)` modules. Old checkpoints saved with the loop layout migrate via `scripts/migrate_multihead_checkpoint.py`:

```bash
python scripts/migrate_multihead_checkpoint.py \
    --config configs/extensions/multihead_pk.yaml \
    --checkpoint old_ckpt.pt \
    --output    old_ckpt.migrated.pt
```

The migration is bit-exact (cat along dim 0 of the H per-head weight tensors; same cat applied to `exp_avg` and `exp_avg_sq` Adam moments), so resuming from `migrated.pt` is mathematically identical to "if the old code had finished training."

## Hardware notes

This study used Princeton Adroit (4× A100 80 GB on the GPU node). Per-cell wall times ~16 h on a single A100 80 GB at the default `batch_size: 12 × gradient_accumulation_steps: 3` (effective batch 36). On 40 GB GPUs, drop to `batch_size: 6` and `gradient_accumulation_steps: 6` to keep the effective batch matched.

Distillation runs are ~1.3× wall (extra teacher forward per step) so the Slurm wall is set to 36 h in `scripts/slurm/distill.sbatch`.

The per-layer routing analysis is fully CPU; the LoRA residual is tiny vs the LLaMA backbone forward at low batch size, so a CPU walk is ~30 min for 6 K=64 checkpoints.
