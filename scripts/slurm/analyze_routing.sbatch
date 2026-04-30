#!/usr/bin/env bash
#SBATCH --job-name=moe-lora-analyze
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/logs/slurm-%x-%j.out
#SBATCH --error=results/logs/slurm-%x-%j.err

# CPU-only analysis: walks the trained K=64 checkpoints and emits per-module
# routing JSON + a cross-checkpoint summary. No GPU required (the LoRA
# residual is tiny vs the LLaMA backbone forward at low batch size).
#
# Usage:
#   sbatch scripts/slurm/analyze_routing.sbatch
#   # Or specify which models to analyze:
#   DEMO_MODELS=0,1,2 DEMO_SAMPLES=10 sbatch scripts/slurm/analyze_routing.sbatch

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
ACTIVATE="${ACTIVATE:-$REPO_ROOT/venv/bin/activate}"
[ -f "$ACTIVATE" ] && source "$ACTIVATE"
cd "$REPO_ROOT"

mkdir -p results/analysis

# Per-module top-k routing across datasets
python -m scalable_moe_lora.analysis.per_layer_routing

# Per-module gate magnitudes (sharpness statistics)
python -m scalable_moe_lora.analysis.gate_magnitudes

# Cross-checkpoint summary (entropy, dead-expert count, hot-set Jaccard)
python -m scalable_moe_lora.analysis.per_layer_summary
