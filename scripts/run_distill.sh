#!/usr/bin/env bash
# Local launcher for a distillation training run. The student is trained with
# the standard task loss + load-balance penalty + a KL term that pushes the
# student's K-wide router softmax toward the teacher's at every MoE-LoRA
# module.
#
# Usage:
#   ./scripts/run_distill.sh <student-config> <teacher-config> <teacher-ckpt> [seed] [distill_coef]
#
# Example (product-key student distilled from K=64 linear-router teacher):
#   ./scripts/run_distill.sh \
#       configs/extensions/distill_pk_from_linear.yaml \
#       configs/c_routers/linear.yaml \
#       results/checkpoints/routers_linear_..._best.pt \
#       42 1.0

set -euo pipefail

STUDENT_CFG="${1:-}"; TEACHER_CFG="${2:-}"; TEACHER_CKPT="${3:-}"
[ -n "$STUDENT_CFG" ] && [ -n "$TEACHER_CFG" ] && [ -n "$TEACHER_CKPT" ] || {
    echo "usage: $0 <student-config> <teacher-config> <teacher-ckpt> [seed] [distill_coef]" >&2
    exit 1; }
SEED="${4:-42}"
DISTILL_COEF="${5:-1.0}"

DATASETS="${DATASETS:-gsm8k,arc,commonsenseqa,piqa,winogrande,boolq,hellaswag,math,openbookqa,sciq,mbpp,logiqa2,drop,mmlu_aux,triviaqa,anli,e2e,samsum}"

python -m scalable_moe_lora.train_reasoning \
    --config "$STUDENT_CFG" \
    --datasets "$DATASETS" \
    --seed "$SEED" \
    --teacher_config "$TEACHER_CFG" \
    --teacher_ckpt "$TEACHER_CKPT" \
    --distill_coef "$DISTILL_COEF"
