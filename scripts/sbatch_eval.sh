#!/bin/bash
#SBATCH --job-name=cos484-eval
#SBATCH --partition=gpu
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --nodelist=adroit-h11g1
#SBATCH --time=03:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@princeton.edu
#SBATCH --output=${PROJECT_ROOT}/results/runs/logs/slurm-%x-%j.out
#SBATCH --error=${PROJECT_ROOT}/results/runs/logs/slurm-%x-%j.err

# Eval-only sbatch wrapper (Phase E). Runs evaluate_reasoning.py on a
# pre-trained checkpoint, reporting both in-distribution accuracy (training
# datasets) and OOD accuracy on held-out benchmarks (MMLU, MMLU-Pro, BBH,
# IFEval, AGIEval, GPQA Diamond, TruthfulQA).
#
# Usage:
#   CONFIG=granularity_r1_k64 \
#   CKPT=results/runs/checkpoints/llama_granularity_r1_k64_..._best.pt \
#   OUT_NAME=llama_granularity_r1_k64 \
#   SEED=42 \
#     sbatch scripts/sbatch_eval_only.sh

set -euo pipefail

: "${CONFIG:?Set CONFIG=<config_name_without_yaml>}"
: "${CKPT:?Set CKPT=<absolute path to _best.pt>}"

# If CKPT points at a missing _best.pt, fall back to the sibling _final.pt.
# This covers the case where val_loss tracking never fired (e.g., NaN val)
# and training produced only _final.pt, not _best.pt.
if [ ! -f "$CKPT" ] && [[ "$CKPT" == *_best.pt ]]; then
    CKPT_FALLBACK="${CKPT%_best.pt}_final.pt"
    if [ -f "$CKPT_FALLBACK" ]; then
        echo ">> _best.pt missing, falling back to _final.pt: $CKPT_FALLBACK"
        CKPT="$CKPT_FALLBACK"
    fi
fi

SEED="${SEED:-42}"
DATASETS="${DATASETS:-gsm8k,arc,commonsenseqa,piqa,winogrande,boolq,hellaswag,math,openbookqa,sciq,mbpp,logiqa2,drop,mmlu_aux,triviaqa,anli,e2e,samsum}"
EVAL_DATASETS="${EVAL_DATASETS:-mmlu,mmlu_pro,bbh,ifeval,agieval,gpqa_diamond,truthfulqa}"
ROUTING="${ROUTING:-no}"

export PROJECT_ROOT="${PROJECT_ROOT}"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/env.sh"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CONFIG_YAML="$PROJECT_ROOT/configs/${CONFIG}.yaml"
[ -f "$CONFIG_YAML" ] || { echo "ERROR: config not found: $CONFIG_YAML"; exit 2; }
[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT (and no _final.pt fallback)"; exit 2; }

if [ -z "${OUT_NAME:-}" ]; then
    OUT_NAME=$(grep -E '^name:' "$CONFIG_YAML" | head -1 | awk '{print $2}')
fi
DS_TAG=$(echo "$DATASETS" | tr ',' '+')
EVAL_TAG=$(echo "$EVAL_DATASETS" | tr ',' '+')

RESULTS_DIR="$PROJECT_ROOT/results/runs"
OUT_FILE="$RESULTS_DIR/logs/${OUT_NAME}_${DS_TAG}_seed${SEED}_phase_e_${EVAL_TAG}.json"
ERROR_LOG="$RESULTS_DIR/logs/${OUT_NAME}_seed${SEED}_phase_e_error.log"

mkdir -p "$RESULTS_DIR/logs"

echo "=========================================="
echo "  Phase E eval job: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME"
echo "  Config: $CONFIG_YAML"
echo "  Checkpoint: $CKPT"
echo "  In-dist datasets: $DATASETS"
echo "  Eval-only (OOD): $EVAL_DATASETS"
echo "  Started: $(date)"
echo "=========================================="

nvidia-smi

ROUTING_FLAG=""
if [ "$ROUTING" = "yes" ]; then
    ROUTING_FLAG="--routing_analysis"
fi

python3 src/evaluate_reasoning.py \
    --config "$CONFIG_YAML" \
    --checkpoint "$CKPT" \
    --datasets "$DATASETS" \
    --eval_datasets "$EVAL_DATASETS" \
    --seed "$SEED" \
    --output "$OUT_FILE" \
    $ROUTING_FLAG 2>&1 | tee -a "$ERROR_LOG"

echo ""
echo "=========================================="
echo "  Finished: $(date)"
echo "  Output: $OUT_FILE"
echo "=========================================="
