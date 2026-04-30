"""Evaluation for reasoning + NLG tasks."""

import torch
torch.cuda.empty_cache()

import argparse
import json
import math
import os
import re

from scalable_moe_lora.evaluate import generate_predictions, compute_metrics as compute_nlg_metrics
from scalable_moe_lora.model import build_model
from scalable_moe_lora.data.reasoning import REASONING_DATASET_LOADERS, load_raw_dataset
from scalable_moe_lora.utils import load_config, load_checkpoint, set_seed
from scalable_moe_lora.adapters import MoELoRA


def extract_gsm8k_answer(text):
    match = re.search(r'####\s*([+-]?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = re.findall(r'[+-]?[\d,]+\.?\d*', text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return ""


def extract_mcq_letter(text):
    text = text.strip()
    if text and text[0].upper() in "ABCDEFGHIJ":
        return text[0].upper()
    match = re.search(r'\b([A-J])\b', text)
    if match:
        return match.group(1)
    return ""


def extract_exact_match_normalize(text):
    """Normalized exact-match for BBH: strips surrounding non-word chars
    (including parens) so "(A)" and "A" both normalize to "a"."""
    if not text:
        return ""
    text = text.strip().split("\n")[0]
    text = re.sub(r"^[^\w]+|[^\w]+$", "", text)
    return text.lower()


def extract_mcq_digit(text):
    text = text.strip()
    if text and text[0] in "1234":
        return text[0]
    match = re.search(r'\b([1-4])\b', text)
    if match:
        return match.group(1)
    return ""


def extract_yesno(text):
    text = text.strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    if "yes" in text:
        return "yes"
    if "no" in text:
        return "no"
    return ""


def extract_math_boxed(text):
    if not text:
        return ""
    pattern = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    if nums:
        return nums[-1]
    return ""


def extract_drop_span(text):
    if not text:
        return ""
    first_line = text.strip().split("\n")[0].strip()
    first_line = first_line.rstrip(".,;:!?")
    return first_line


def extract_factoid_normalize(text):
    if not text:
        return ""
    text = text.strip().split("\n")[0].strip()
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


EXTRACTORS = {
    "gsm8k": extract_gsm8k_answer,
    "mcq_letter": extract_mcq_letter,
    "mcq_digit": extract_mcq_digit,
    "yesno": extract_yesno,
    "math_boxed": extract_math_boxed,
    "drop_span": extract_drop_span,
    "factoid_normalize": extract_factoid_normalize,
    "exact_match_normalize": extract_exact_match_normalize,
}


DEFAULT_MAX_NEW_TOKENS = 64
MAX_NEW_TOKENS_BY_DATASET = {
    "gsm8k": 256,
    "arc": 8,
    "commonsenseqa": 8,
    "piqa": 8,
    "winogrande": 8,
    "boolq": 8,
    "hellaswag": 8,
    "e2e": 64,
    "samsum": 128,
    "math": 384,
    "openbookqa": 8,
    "sciq": 8,
    "mbpp": 256,
    "logiqa2": 8,
    "drop": 32,
    "mmlu_aux": 8,
    "triviaqa": 32,
    "anli": 8,
    "mmlu": 8,
    "mmlu_pro": 8,
    "bbh": 128,
    "ifeval": 256,
    "agieval": 8,
    "gpqa_diamond": 8,
    "truthfulqa": 8,
}


def compute_accuracy(predictions, references, extractor_name):
    extractor = EXTRACTORS[extractor_name]
    correct = 0
    total = len(predictions)
    for pred, ref in zip(predictions, references):
        extracted_pred = extractor(pred)
        extracted_ref = extractor(str(ref))
        if extractor_name == "gsm8k":
            try:
                if float(extracted_pred) == float(extracted_ref):
                    correct += 1
            except (ValueError, TypeError):
                pass
        else:
            if extracted_pred and extracted_pred.lower() == extracted_ref.lower():
                correct += 1
    return correct / max(total, 1)


def compute_dataset_metrics(predictions, references, dataset_name):
    info = REASONING_DATASET_LOADERS[dataset_name]

    if info["metric"] == "accuracy":
        acc = compute_accuracy(predictions, references, info["answer_extractor"])
        return {"accuracy": acc}
    else:
        return compute_nlg_metrics(predictions, references)


def collect_routing_stats(model, tokenizer, inputs, device, max_samples=200):
    model.eval()
    tokenizer.padding_side = "left"
    eos_token = tokenizer.eos_token or "</s>"

    routing_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, MoELoRA):
            routing_modules[name] = module

    if not routing_modules:
        return {}

    all_indices = {name: [] for name in routing_modules}
    inputs = inputs[:max_samples]

    with torch.no_grad():
        batch_size = 16
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            prompts = [f"{inp}{eos_token}" for inp in batch]
            encoded = tokenizer(
                prompts, return_tensors="pt", truncation=True,
                max_length=384, padding=True,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            model(input_ids=input_ids, attention_mask=attention_mask)

            for name, module in routing_modules.items():
                if hasattr(module, "_last_routing_indices"):
                    mask = attention_mask.bool()
                    indices = module._last_routing_indices
                    for b in range(indices.shape[0]):
                        valid_len = mask[b].sum().item()
                        all_indices[name].append(indices[b, :valid_len].cpu())

    result = {}
    for name, idx_list in all_indices.items():
        if idx_list:
            result[name] = torch.cat(idx_list, dim=0)
    return result


def compute_routing_analysis(routing_indices, num_components):
    if not routing_indices:
        return {}

    all_counts = torch.zeros(num_components)
    for name, indices in routing_indices.items():
        flat = indices.flatten()
        for idx in flat:
            all_counts[idx.item()] += 1

    total = all_counts.sum().item()
    if total == 0:
        return {}

    probs = all_counts / total
    entropy = -sum(p.item() * math.log2(p.item()) for p in probs if p.item() > 0)
    max_entropy = math.log2(num_components)

    used = (all_counts > 0).sum().item()
    used_1pct = (probs > 0.01).sum().item()

    top_k_vals, top_k_ids = torch.topk(all_counts, min(5, num_components))

    return {
        "entropy": entropy,
        "max_entropy": max_entropy,
        "normalized_entropy": entropy / max_entropy if max_entropy > 0 else 0,
        "components_used": used,
        "components_used_1pct": used_1pct,
        "total_components": num_components,
        "top_components": {int(k): int(v) for k, v in zip(top_k_ids, top_k_vals)},
        "usage_counts": all_counts.tolist(),
    }


def evaluate_all_datasets(config_path, checkpoint_path, dataset_names, seed=42,
                          routing_analysis=False, eval_datasets=None):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(config_path)
    model, tokenizer = build_model(config)
    if checkpoint_path:
        load_checkpoint(checkpoint_path, model)
    else:
        # Zero-shot mode: no adapter loaded. LoRA modules are still injected
        # but their B weights are zero at init, so the forward pass is a
        # mathematical no-op over the base model.
        print("  [zero-shot] no checkpoint provided; evaluating base model "
              "with zero-initialized LoRA branches (mathematical no-op).")
    model = model.to(device)

    all_metrics = {}
    routing_by_dataset = {}

    full_eval_list = list(dataset_names)
    if eval_datasets:
        full_eval_list.extend(eval_datasets)

    for ds_name in full_eval_list:
        print(f"\n--- Evaluating on {ds_name} ---")
        inputs, references = load_raw_dataset(ds_name, split="test")

        max_new = MAX_NEW_TOKENS_BY_DATASET.get(ds_name, DEFAULT_MAX_NEW_TOKENS)
        predictions = generate_predictions(
            model, tokenizer, inputs, device, max_new_tokens=max_new
        )

        metrics = compute_dataset_metrics(predictions, references, ds_name)
        all_metrics[ds_name] = metrics

        info = REASONING_DATASET_LOADERS[ds_name]
        if info["metric"] == "accuracy":
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
        else:
            print(f"  BLEU: {metrics['bleu']:.4f} | ROUGE-L: {metrics['rougeL']:.4f}")

        if routing_analysis:
            print(f"  Collecting routing stats for {ds_name}...")
            indices = collect_routing_stats(model, tokenizer, inputs, device)
            if indices:
                first_layer = next(
                    (m for m in model.modules() if isinstance(m, MoELoRA)),
                    None,
                )
                num_comp = first_layer.num_experts if first_layer else 0

                if num_comp > 0:
                    stats = compute_routing_analysis(indices, num_comp)
                    routing_by_dataset[ds_name] = stats
                    print(f"  Routing entropy: {stats['entropy']:.2f} / {stats['max_entropy']:.2f} "
                          f"({stats['normalized_entropy']:.2f} normalized)")
                    print(f"  Components used (>1%): {stats['components_used_1pct']}/{stats['total_components']}")

    return all_metrics, routing_by_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--datasets", type=str, required=True,
                        help="In-distribution training datasets to evaluate on")
    parser.add_argument("--eval_datasets", type=str, default=None,
                        help="Optional held-out OOD eval benchmarks (comma-separated)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--routing_analysis", action="store_true")
    args = parser.parse_args()

    dataset_names = args.datasets.split(",")
    eval_datasets = args.eval_datasets.split(",") if args.eval_datasets else None
    all_metrics, routing_stats = evaluate_all_datasets(
        args.config, args.checkpoint, dataset_names, args.seed,
        routing_analysis=args.routing_analysis,
        eval_datasets=eval_datasets,
    )

    results = {"metrics": all_metrics}
    if routing_stats:
        results["routing"] = routing_stats

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
