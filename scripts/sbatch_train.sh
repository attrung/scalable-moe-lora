#!/bin/bash
#SBATCH --job-name=cos484-lora-m
#SBATCH --partition=gpu
#SBATCH --qos=gpu-medium
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --nodelist=adroit-h11g1
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@princeton.edu
#SBATCH --output=${PROJECT_ROOT}/results/runs/logs/slurm-%x-%j.out
#SBATCH --error=${PROJECT_ROOT}/results/runs/logs/slurm-%x-%j.err

# PRIMARY v2 training sbatch wrapper (gpu-medium QoS, 14 h wall).
#
# Note: after the v2 504k-example suite, per-variant training time is ~11 h
# (47274 optimizer steps at ~0.82 sec/step). This exceeds gpu-short's 4 h cap
# which will TIMEOUT jobs and LOSE all progress (epoch 1 is ~15k steps and
# finishes at ~3:45, often right at the wall-time cliff — no checkpoint saved).
#
# gpu-medium has 2-concurrent GPU cap per user and priority 10. Wall time
# allowed up to 24 h; we request 14 h to cover the ~11 h training + ~1 h eval
# + buffer.
#
# Same interface as sbatch_variant.sh — set CONFIG, SEED, DATASETS, OUT_NAME,
# ROUTING, MAX_STEPS, SKIP_EVAL env vars as needed.

set -euo pipefail

: "${CONFIG:?Set CONFIG=<config_name_without_yaml>}"
SEED="${SEED:-42}"
DATASETS="${DATASETS:-gsm8k,arc,commonsenseqa,piqa,winogrande,boolq,hellaswag,math,openbookqa,sciq,mbpp,logiqa2,drop,mmlu_aux,triviaqa,anli,e2e,samsum}"
ROUTING="${ROUTING:-no}"
MAX_STEPS="${MAX_STEPS:-}"
SKIP_EVAL="${SKIP_EVAL:-no}"
SMOKE_WITH_VAL="${SMOKE_WITH_VAL:-no}"

export PROJECT_ROOT="${PROJECT_ROOT}"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/env.sh"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CONFIG_YAML="$PROJECT_ROOT/configs/${CONFIG}.yaml"
[ -f "$CONFIG_YAML" ] || { echo "ERROR: config not found: $CONFIG_YAML"; exit 2; }

if [ -z "${OUT_NAME:-}" ]; then
    OUT_NAME=$(grep -E '^name:' "$CONFIG_YAML" | head -1 | awk '{print $2}')
fi
DS_TAG=$(echo "$DATASETS" | tr ',' '+')

RESULTS_DIR="$PROJECT_ROOT/results/runs"
CKPT_BEST="$RESULTS_DIR/checkpoints/${OUT_NAME}_${DS_TAG}_seed${SEED}_best.pt"
CKPT_FINAL="$RESULTS_DIR/checkpoints/${OUT_NAME}_${DS_TAG}_seed${SEED}_final.pt"
EVAL_JSON="$RESULTS_DIR/logs/${OUT_NAME}_${DS_TAG}_seed${SEED}_eval.json"
ERROR_LOG="$RESULTS_DIR/logs/${OUT_NAME}_${DS_TAG}_seed${SEED}_error.log"

mkdir -p "$RESULTS_DIR/checkpoints" "$RESULTS_DIR/logs"

echo "=========================================="
echo "  Job: $SLURM_JOB_ID ($SLURM_JOB_NAME) [medium QoS fallback]"
echo "  Node: $SLURMD_NODENAME"
echo "  Config: $CONFIG_YAML"
echo "  Started: $(date)"
echo "=========================================="

nvidia-smi

MAX_STEPS_ARG=""
if [ -n "$MAX_STEPS" ]; then
    MAX_STEPS_ARG="--max_steps $MAX_STEPS"
fi

SMOKE_WITH_VAL_ARG=""
if [ "$SMOKE_WITH_VAL" = "yes" ]; then
    SMOKE_WITH_VAL_ARG="--smoke_with_val"
fi

if [ -f "$CKPT_BEST" ] && [ -f "$CKPT_FINAL" ] && [ -z "$MAX_STEPS" ]; then
    echo ">> Checkpoints already exist — skipping training"
else
    echo ""
    echo "=== TRAINING ==="
    python3 src/train_reasoning.py \
        --config "$CONFIG_YAML" \
        --datasets "$DATASETS" \
        --seed "$SEED" \
        $MAX_STEPS_ARG $SMOKE_WITH_VAL_ARG 2>&1 | tee -a "$ERROR_LOG"
fi

if [ "$SKIP_EVAL" = "yes" ]; then
    echo ">> SKIP_EVAL=yes — skipping eval"
elif [ -f "$EVAL_JSON" ]; then
    echo ">> Eval already exists — skipping"
else
    echo ""
    echo "=== EVAL ==="

    CKPT_TO_EVAL="$CKPT_BEST"
    if [ ! -f "$CKPT_TO_EVAL" ]; then
        if [ -f "$CKPT_FINAL" ]; then
            echo ">> _best.pt missing, falling back to _final.pt"
            CKPT_TO_EVAL="$CKPT_FINAL"
        else
            echo "ERROR: neither _best.pt nor _final.pt exists — cannot run eval"
            exit 3
        fi
    fi

    ROUTING_FLAG=""
    if [ "$ROUTING" = "yes" ]; then
        ROUTING_FLAG="--routing_analysis"
    fi
    python3 src/evaluate_reasoning.py \
        --config "$CONFIG_YAML" \
        --checkpoint "$CKPT_TO_EVAL" \
        --datasets "$DATASETS" \
        --seed "$SEED" \
        --output "$EVAL_JSON" \
        $ROUTING_FLAG 2>&1 | tee -a "$ERROR_LOG"
fi

echo ""
echo "=========================================="
echo "  Finished: $(date)"
echo "=========================================="
