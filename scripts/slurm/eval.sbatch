#!/usr/bin/env bash
#SBATCH --job-name=moe-lora-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/logs/slurm-%x-%j.out
#SBATCH --error=results/logs/slurm-%x-%j.err

# Slurm wrapper for in-distribution + (optional) OOD evaluation on a saved
# checkpoint. Generation-based eval, single GPU, ~1-2 h depending on dataset
# count and beam width.
#
# Usage:
#   CONFIG=configs/c_routers/linear.yaml \\
#       CKPT=results/checkpoints/routers_linear_..._best.pt \\
#       SEED=42 \\
#       OOD=yes \\
#       sbatch scripts/slurm/eval.sbatch

set -euo pipefail

: "${CONFIG:?Set CONFIG=<yaml>}"
: "${CKPT:?Set CKPT=<checkpoint path>}"
SEED="${SEED:-42}"
DATASETS="${DATASETS:-gsm8k,arc,commonsenseqa,piqa,winogrande,boolq,hellaswag,math,openbookqa,sciq,mbpp,logiqa2,drop,mmlu_aux,triviaqa,anli,e2e,samsum}"
OOD_DATASETS="${OOD_DATASETS:-mmlu,mmlu_pro,bbh,ifeval,agieval,gpqa_diamond,truthfulqa}"
OOD="${OOD:-no}"
ROUTING="${ROUTING:-no}"

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
ACTIVATE="${ACTIVATE:-$REPO_ROOT/venv/bin/activate}"
[ -f "$ACTIVATE" ] && source "$ACTIVATE"
cd "$REPO_ROOT"

OUTPUT="${OUTPUT:-${CKPT%.pt}_eval.json}"
EXTRA=""
[ "$OOD" = "yes" ]     && EXTRA="$EXTRA --eval_datasets $OOD_DATASETS"
[ "$ROUTING" = "yes" ] && EXTRA="$EXTRA --routing_analysis"

echo "================================================================"
echo "  Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "  Config: $CONFIG"
echo "  Checkpoint: $CKPT"
echo "  Output: $OUTPUT"
echo "  OOD: $OOD   Routing-analysis: $ROUTING"
echo "================================================================"

python -m scalable_moe_lora.evaluate_reasoning \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --datasets "$DATASETS" \
    --seed "$SEED" \
    --output "$OUTPUT" \
    $EXTRA
