#!/usr/bin/env bash
#SBATCH --job-name=moe-lora-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/logs/slurm-%x-%j.out
#SBATCH --error=results/logs/slurm-%x-%j.err

# Slurm wrapper for training. Override partition / QoS / constraint flags
# above to match your cluster (e.g. for an A100-only constraint:
# `#SBATCH --constraint=gpu80`).
#
# Usage:
#   CONFIG=configs/c_routers/linear.yaml SEED=42 \\
#       sbatch scripts/slurm/train.sbatch
#
#   # Resume from a checkpoint (e.g. after a wall-time-out):
#   CONFIG=configs/extensions/multihead_pk.yaml SEED=42 \\
#       RESUME_FROM=results/checkpoints/<...>.migrated.pt \\
#       sbatch scripts/slurm/train.sbatch

set -euo pipefail

: "${CONFIG:?Set CONFIG=<path-to-yaml>}"
SEED="${SEED:-42}"
DATASETS="${DATASETS:-gsm8k,arc,commonsenseqa,piqa,winogrande,boolq,hellaswag,math,openbookqa,sciq,mbpp,logiqa2,drop,mmlu_aux,triviaqa,anli,e2e,samsum}"
RESUME_FROM="${RESUME_FROM:-}"

# Activate your environment (override REPO_ROOT and ACTIVATE if needed).
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
ACTIVATE="${ACTIVATE:-$REPO_ROOT/venv/bin/activate}"
[ -f "$ACTIVATE" ] && source "$ACTIVATE"

cd "$REPO_ROOT"
mkdir -p results/{logs,checkpoints}

# PYTHONHASHSEED must be set before the python process starts; in-process
# os.environ assignment is too late (Python's hash randomization is fixed at
# interpreter start). Affects sciq/gpqa_diamond option placement.
export PYTHONHASHSEED=0

EXTRA=""
[ -n "$RESUME_FROM" ] && EXTRA="--resume_from $RESUME_FROM"

echo "================================================================"
echo "  Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "  Config: $CONFIG"
echo "  Seed: $SEED"
[ -n "$RESUME_FROM" ] && echo "  Resume: $RESUME_FROM"
echo "  Started: $(date)"
echo "================================================================"

python -m scalable_moe_lora.train_reasoning \
    --config "$CONFIG" \
    --datasets "$DATASETS" \
    --seed "$SEED" \
    $EXTRA
