#!/usr/bin/env bash
#SBATCH --job-name=moe-lora-distill
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/logs/slurm-%x-%j.out
#SBATCH --error=results/logs/slurm-%x-%j.err

# Slurm wrapper for distillation training. The student is trained with the
# standard task loss + load-balance penalty + a KL term that pushes the
# student's K-wide router softmax toward the teacher's at every RoutedLoRA
# module.
#
# Wall is set to 36 h (vs 24 h for plain training) because the per-step cost
# is ~1.3x with the extra teacher forward.
#
# Usage:
#   STUDENT_CONFIG=configs/extensions/distill_pk_from_linear.yaml \\
#       TEACHER_CONFIG=configs/c_routers/linear.yaml \\
#       TEACHER_CKPT=results/checkpoints/routers_linear_..._best.pt \\
#       DISTILL_COEF=1.0 SEED=42 \\
#       sbatch scripts/slurm/distill.sbatch

set -euo pipefail

: "${STUDENT_CONFIG:?Set STUDENT_CONFIG=<student yaml>}"
: "${TEACHER_CONFIG:?Set TEACHER_CONFIG=<teacher yaml>}"
: "${TEACHER_CKPT:?Set TEACHER_CKPT=<teacher checkpoint .pt>}"
DISTILL_COEF="${DISTILL_COEF:-1.0}"
SEED="${SEED:-42}"
DATASETS="${DATASETS:-gsm8k,arc,commonsenseqa,piqa,winogrande,boolq,hellaswag,math,openbookqa,sciq,mbpp,logiqa2,drop,mmlu_aux,triviaqa,anli,e2e,samsum}"

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
ACTIVATE="${ACTIVATE:-$REPO_ROOT/venv/bin/activate}"
[ -f "$ACTIVATE" ] && source "$ACTIVATE"
cd "$REPO_ROOT"

[ -f "$TEACHER_CONFIG" ] || { echo "ERROR: teacher_config not found: $TEACHER_CONFIG" >&2; exit 2; }
[ -f "$TEACHER_CKPT" ]  || { echo "ERROR: teacher_ckpt not found: $TEACHER_CKPT" >&2; exit 2; }

echo "================================================================"
echo "  Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "  Student: $STUDENT_CONFIG"
echo "  Teacher: $TEACHER_CONFIG"
echo "    ckpt:  $TEACHER_CKPT"
echo "  distill_coef: $DISTILL_COEF"
echo "================================================================"

python -m scalable_moe_lora.train_reasoning \
    --config "$STUDENT_CONFIG" \
    --datasets "$DATASETS" \
    --seed "$SEED" \
    --teacher_config "$TEACHER_CONFIG" \
    --teacher_ckpt "$TEACHER_CKPT" \
    --distill_coef "$DISTILL_COEF"
