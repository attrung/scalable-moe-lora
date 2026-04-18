"""Evaluation for LLaMA: generate text and compute BLEU/ROUGE-L."""

import torch
torch.cuda.empty_cache()

import argparse
import json
import os
import nltk
import importlib, sys

# Load the HuggingFace `evaluate` package, guarding against shadowing by
# our own src/evaluate.py when running as a script from the repo root.
_orig_path = sys.path[:]
sys.path = [p for p in sys.path if os.path.basename(p.rstrip("/")) != "src"]
hf_evaluate = importlib.import_module("evaluate")
sys.path = _orig_path

from src.model import build_model
from src.utils import load_config, load_checkpoint, set_seed
from src.data import load_raw_dataset

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def generate_predictions(model, tokenizer, inputs, device, max_new_tokens=64, beam_size=1, batch_size=32):
    model.eval()
    predictions = []
    tokenizer.padding_side = "left"

    eos_token = tokenizer.eos_token or "</s>"
    prompts = [f"{inp}{eos_token}" for inp in inputs]

    from tqdm import tqdm
    progress_bar = tqdm(total=len(prompts), desc="Generating")

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            encoded = tokenizer(
                batch_prompts, return_tensors="pt", truncation=True,
                max_length=384, padding=True,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            prompt_len = input_ids.shape[1]

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=beam_size,
                no_repeat_ngram_size=4,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            for j in range(len(batch_prompts)):
                generated = outputs[j][prompt_len:]
                if len(generated) == 0:
                    text = ""
                else:
                    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
                predictions.append(text)

            progress_bar.update(len(batch_prompts))

    progress_bar.close()
    return predictions


def compute_metrics(predictions, references):
    """Compute corpus BLEU-4 + mean ROUGE-L.

    Uses nltk.translate.bleu_score and rouge_score directly instead of the
    HuggingFace `evaluate` library. The evaluate library's `load("bleu")`
    tries to fetch a module script from the Hub which fails on offline
    compute nodes (baseline v1 crashed this way). nltk is pip-installed
    as part of the venv and has no network dependency.
    """
    # Normalize references to always be list-of-lists (multi-reference support)
    if references and isinstance(references[0], str):
        references = [[ref] for ref in references]

    # ---- BLEU via nltk ----
    # Tokenize: lowercase + whitespace split (simple, matches hf_evaluate's
    # default wordpiece tokenizer behaviour reasonably well for our purposes).
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    def _tok(s):
        return s.strip().lower().split()

    tokenized_preds = [_tok(p) for p in predictions]
    tokenized_refs = [[_tok(r) for r in refs] for refs in references]

    # Use SmoothingFunction method1 to avoid zero-BLEU on short sentences
    smoother = SmoothingFunction().method1
    try:
        bleu_score = corpus_bleu(
            tokenized_refs, tokenized_preds,
            smoothing_function=smoother,
        )
    except ZeroDivisionError:
        bleu_score = 0.0

    # ---- ROUGE-L via rouge_score library ----
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    best_rouge_scores = []
    for pred, refs in zip(predictions, references):
        best = max(scorer.score(ref, pred)["rougeL"].fmeasure for ref in refs)
        best_rouge_scores.append(best)
    rouge_l = sum(best_rouge_scores) / len(best_rouge_scores)

    return {
        "bleu": bleu_score,
        "rougeL": rouge_l,
    }


def evaluate_all_datasets(config_path, checkpoint_path, dataset_names, seed=42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(config_path)
    model, tokenizer = build_model(config)
    load_checkpoint(checkpoint_path, model)
    model = model.to(device)

    all_metrics = {}
    for ds_name in dataset_names:
        print(f"\n--- Evaluating on {ds_name} ---")
        inputs, references = load_raw_dataset(ds_name, split="test")
        predictions = generate_predictions(model, tokenizer, inputs, device)
        metrics = compute_metrics(predictions, references)
        all_metrics[ds_name] = metrics
        print(f"  BLEU: {metrics['bleu']:.4f} | ROUGE-L: {metrics['rougeL']:.4f}")

    return all_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--datasets", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    dataset_names = args.datasets.split(",")
    all_metrics = evaluate_all_datasets(
        args.config, args.checkpoint, dataset_names, args.seed
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"metrics": all_metrics}, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
