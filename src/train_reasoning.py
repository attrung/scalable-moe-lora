"""Training wrapper for reasoning + NLG experiments.

Monkey-patches train.py to use reasoning data loaders and mixed eval metrics
(accuracy for reasoning datasets, BLEU/ROUGE-L for NLG datasets).
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.cuda.empty_cache()

import sys
import time

import src.train as train_module
from src.data import make_collate_fn
from src.data_reasoning import (
    load_multitask_dataset,
    REASONING_DATASET_LOADERS,
)

train_module.load_multitask_dataset = load_multitask_dataset
train_module.make_collate_fn = make_collate_fn


def quick_eval_reasoning(model, tokenizer, device, dataset_name, max_samples=100):
    """Quick eval that dispatches accuracy vs BLEU/ROUGE-L based on dataset type."""
    try:
        from src.evaluate import generate_predictions
        from src.data_reasoning import load_raw_dataset
        from src.evaluate_reasoning import compute_dataset_metrics

        inputs, references = load_raw_dataset(dataset_name, split="test")
        inputs = inputs[:max_samples]
        references = references[:max_samples]
        predictions = generate_predictions(
            model, tokenizer, inputs, device, max_new_tokens=64, beam_size=1
        )
        metrics = compute_dataset_metrics(predictions, references, dataset_name)

        info = REASONING_DATASET_LOADERS.get(dataset_name, {})
        if info.get("metric") == "accuracy":
            # Return in a format the epoch summary can display
            return {"accuracy": metrics["accuracy"], "bleu": -1, "rougeL": -1}
        return metrics
    except Exception as e:
        return {"bleu": -1, "rougeL": -1, "accuracy": -1, "error": str(e)}


# Patch quick_eval
train_module.quick_eval = quick_eval_reasoning

# Patch the epoch summary format to show accuracy when available
_original_train = train_module.train


def train_with_reasoning_summary(config, dataset_names, seed, max_steps=None, smoke_with_val=False):
    return _original_train(config, dataset_names, seed, max_steps, smoke_with_val=smoke_with_val)


train_module.train = train_with_reasoning_summary

RESULTS_DIR = train_module.RESULTS_DIR


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--datasets", type=str, required=True,
                        help="Comma-separated training datasets, e.g. gsm8k,arc,mbpp,logiqa2,drop,...")
    parser.add_argument("--eval_datasets", type=str, default=None,
                        help="Optional held-out eval-only benchmarks (unused during training; reserved "
                             "for Phase E eval-only runs that reuse this wrapper's argparse)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--smoke_with_val", action="store_true",
                        help="When --max_steps is set, still run validate() + best-tracking "
                             "+ inline eval at the end of the smoke run (short end-to-end "
                             "pipeline check instead of the default skip-validation smoke).")
    args = parser.parse_args()

    config = train_module.load_config(args.config)
    dataset_names = args.datasets.split(",")
    train_module.train(config, dataset_names, args.seed, args.max_steps,
                       smoke_with_val=args.smoke_with_val)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        crash_log = os.path.join(RESULTS_DIR, "logs", "crash.log")
        os.makedirs(os.path.dirname(crash_log), exist_ok=True)
        with open(crash_log, "a") as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"CRASH at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Args: {sys.argv}\n")
            f.write(traceback.format_exc())
        sys.exit(1)
