#!/usr/bin/env bash
# Local launcher for a training run. Works on any machine with a CUDA-capable
# GPU; for Slurm clusters, see scripts/slurm/train.sbatch.
#
# Usage:
#   ./scripts/run_train.sh <config-path> [seed] [extra args...]
#
# Examples:
#   ./scripts/run_train.sh configs/c_routers/linear.yaml
#   ./scripts/run_train.sh configs/c_routers/linear.yaml 42
#   ./scripts/run_train.sh configs/c_routers/linear.yaml 42 --resume_from results/checkpoints/foo_best.pt

set -euo pipefail

CONFIG="${1:-}"
[ -n "$CONFIG" ] || { echo "usage: $0 <config-path> [seed] [extra args...]" >&2; exit 1; }
[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 2; }
SEED="${2:-42}"
shift $(( $# >= 2 ? 2 : 1 ))

DATASETS="${DATASETS:-gsm8k,arc,commonsenseqa,piqa,winogrande,boolq,hellaswag,math,openbookqa,sciq,mbpp,logiqa2,drop,mmlu_aux,triviaqa,anli,e2e,samsum}"

python -m scalable_moe_lora.train_reasoning \
    --config "$CONFIG" \
    --datasets "$DATASETS" \
    --seed "$SEED" \
    "$@"
