#!/usr/bin/env bash
# Local eval launcher. Runs in-distribution eval by default; pass --eval_datasets
# to add the OOD benchmarks (mmlu,mmlu_pro,bbh,ifeval,agieval,gpqa_diamond,truthfulqa).
#
# Usage:
#   ./scripts/run_eval.sh <config-path> <checkpoint-path> [seed] [extra args...]
#
# Examples:
#   In-dist eval:
#     ./scripts/run_eval.sh configs/c_routers/linear.yaml results/checkpoints/.../best.pt
#   In-dist + OOD eval:
#     ./scripts/run_eval.sh configs/c_routers/linear.yaml results/checkpoints/.../best.pt 42 \
#         --eval_datasets mmlu,mmlu_pro,bbh,ifeval,agieval,gpqa_diamond,truthfulqa

set -euo pipefail

CONFIG="${1:-}"; CKPT="${2:-}"
[ -n "$CONFIG" ] && [ -n "$CKPT" ] || {
    echo "usage: $0 <config-path> <checkpoint-path> [seed] [extra args...]" >&2; exit 1; }
[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 2; }
[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT" >&2; exit 2; }
SEED="${3:-42}"
shift $(( $# >= 3 ? 3 : 2 ))

DATASETS="${DATASETS:-gsm8k,arc,commonsenseqa,piqa,winogrande,boolq,hellaswag,math,openbookqa,sciq,mbpp,logiqa2,drop,mmlu_aux,triviaqa,anli,e2e,samsum}"
OUTPUT="${OUTPUT:-${CKPT%.pt}_eval.json}"

python -m scalable_moe_lora.evaluate_reasoning \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --datasets "$DATASETS" \
    --seed "$SEED" \
    --output "$OUTPUT" \
    "$@"
