#!/bin/bash
# Environment setup helper. Source this before running any training or eval.
#
# Expected environment variables (set these to your local paths):
#   PROJECT_ROOT     - absolute path to this repo (default: current dir)
#   VENV_PATH        - path to your Python venv (default: $PROJECT_ROOT/venv)
#   HF_HOME          - HuggingFace cache root (default: $PROJECT_ROOT/hf_cache)

set -e

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/venv}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
fi

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "HF_HOME=$HF_HOME"
echo "VENV=$VENV_PATH"
