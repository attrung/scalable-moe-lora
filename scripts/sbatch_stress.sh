#!/bin/bash
#SBATCH --job-name=cos484-stress
#SBATCH --partition=gpu
#SBATCH --qos=gpu-test
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --nodelist=adroit-h11g1
#SBATCH --time=00:30:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=${USER}@princeton.edu
#SBATCH --output=${PROJECT_ROOT}/results/runs/logs/slurm-stress-%j.out

# VRAM stress test sbatch wrapper.
# Runs vram_stress_test.py under gpu-test QoS (highest priority, immediate
# start) to verify that the given configs fit under the 80% VRAM abort rule
# on A100 80 GB before committing to real training runs.
#
# Invoke with:
#   CONFIGS="reasoning_baseline,granularity_r1_k64,granularity_r2_k32,granularity_r4_k16,granularity_r8_k8" \
#     sbatch scripts/sbatch_stress.sh

set -euo pipefail

: "${CONFIGS:?Set CONFIGS=comma-separated list of config names without .yaml}"

export PROJECT_ROOT="${PROJECT_ROOT}"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/env.sh"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CONFIG_DIR="$PROJECT_ROOT/configs"
CONFIG_PATHS=""
IFS=',' read -ra NAMES <<< "$CONFIGS"
for name in "${NAMES[@]}"; do
    p="$CONFIG_DIR/${name}.yaml"
    [ -f "$p" ] || { echo "ERROR: config not found: $p"; exit 2; }
    if [ -z "$CONFIG_PATHS" ]; then
        CONFIG_PATHS="$p"
    else
        CONFIG_PATHS="$CONFIG_PATHS,$p"
    fi
done

RESULTS_DIR="$PROJECT_ROOT/results/runs"
LOG_FILE="$RESULTS_DIR/logs/stress_test_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$RESULTS_DIR/logs"

echo "=========================================="
echo "  Stress test job: $SLURM_JOB_ID"
echo "  Node: $SLURMD_NODENAME"
echo "  Configs: $CONFIGS"
echo "  Started: $(date)"
echo "=========================================="
nvidia-smi

python3 scripts/vram_stress_test.py \
    --configs "$CONFIG_PATHS" 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "=========================================="
echo "  Finished: $(date)"
echo "  Log: $LOG_FILE"
echo "=========================================="
