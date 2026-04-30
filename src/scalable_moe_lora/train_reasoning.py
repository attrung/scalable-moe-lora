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

import scalable_moe_lora.train as train_module
from scalable_moe_lora.data.nlg import make_collate_fn
from scalable_moe_lora.data.reasoning import (
    load_multitask_dataset,
    REASONING_DATASET_LOADERS,
)

train_module.load_multitask_dataset = load_multitask_dataset
train_module.make_collate_fn = make_collate_fn


def quick_eval_reasoning(model, tokenizer, device, dataset_name, max_samples=100):
    """Quick eval that dispatches accuracy vs BLEU/ROUGE-L based on dataset type."""
    try:
        from scalable_moe_lora.evaluate import generate_predictions
        from scalable_moe_lora.data.reasoning import load_raw_dataset
        from scalable_moe_lora.evaluate_reasoning import compute_dataset_metrics

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


def train_with_reasoning_summary(config, dataset_names, seed, max_steps=None, smoke_with_val=False,
                                  teacher_config=None, teacher_ckpt=None, distill_coef=0.0,
                                  resume_from=None):
    return _original_train(config, dataset_names, seed, max_steps, smoke_with_val=smoke_with_val,
                            teacher_config=teacher_config, teacher_ckpt=teacher_ckpt,
                            distill_coef=distill_coef, resume_from=resume_from)


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
                             "for OOD eval-only runs that reuse this wrapper's argparse)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--smoke_with_val", action="store_true",
                        help="When --max_steps is set, still run validate() + best-tracking "
                             "+ inline eval at the end of the smoke run (short end-to-end "
                             "pipeline check instead of the default skip-validation smoke).")
    parser.add_argument("--teacher_config", type=str, default=None,
                        help="Path to teacher YAML for distillation. Teacher must be a "
                             "RoutedLoRA model with the same K, top_k as the student.")
    parser.add_argument("--teacher_ckpt", type=str, default=None,
                        help="Path to teacher checkpoint (.pt) for distillation.")
    parser.add_argument("--distill_coef", type=float, default=0.0,
                        help="Coefficient on KL(student || teacher) over the K-wide gate "
                             "softmax distribution per RoutedLoRA module. 0 disables.")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a checkpoint (.pt) to resume from. Loads model + "
                             "optimizer state, advances LR scheduler to the saved step, "
                             "and starts the outer loop at the next epoch.")
    args = parser.parse_args()

    config = train_module.load_config(args.config)
    dataset_names = args.datasets.split(",")
    train_module.train(config, dataset_names, args.seed, args.max_steps,
                       smoke_with_val=args.smoke_with_val,
                       teacher_config=args.teacher_config,
                       teacher_ckpt=args.teacher_ckpt,
                       distill_coef=args.distill_coef,
                       resume_from=args.resume_from)


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
