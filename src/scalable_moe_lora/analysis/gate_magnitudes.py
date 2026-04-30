"""Per-layer gate-magnitude analysis for K=64 MoE-LoRA checkpoints (C1).

Tests the §4.4 hypothesis that the linear router's OOD edge comes from
higher-fidelity *soft-gate magnitudes* rather than from sharper top-k
selection. Per module, per dataset, per token, we record the softmax-
normalized gate weights at the top-k positions and compute:

  - mean / std of gate values at top-k (sharpness; higher std = sharper gates)
  - max gate value at top-k (peakiness)
  - normalized entropy H(gate)/log(top_k) (1=uniform across selected,
    0=one expert dominates)
  - top-1 gate fraction = max gate / sum gate (always 1 since softmax,
    but we compute over top-k subset)

The hypothesis predicts the linear router's gate distributions are
sharper / more dataset-dependent than the cheap routers'.

Run on `class` partition (CPU). Reads existing K=64 checkpoints; writes
a single summary JSON to results/routing_per_layer/_gate_magnitudes.json.

    sbatch scripts/sbatch_per_layer_routing.sh \
        --wrap='python3 scripts/per_layer_gate_magnitudes.py'
"""

import sys
import os
import time
import gc
import json
from collections import defaultdict


import torch

from scalable_moe_lora.utils import load_config, load_checkpoint
from scalable_moe_lora.model import build_model
from scalable_moe_lora.data.reasoning import load_raw_dataset
from scalable_moe_lora.adapters import RoutedLoRA


DATASETS = [
    "gsm8k", "arc", "commonsenseqa", "piqa", "winogrande", "boolq",
    "hellaswag", "math", "openbookqa", "sciq", "mbpp", "logiqa2",
    "drop", "mmlu_aux", "triviaqa", "anli", "e2e", "samsum",
]


N_SAMPLES = int(os.environ.get("DEMO_SAMPLES", "5"))
MAX_LEN = int(os.environ.get("DEMO_MAX_LEN", "128"))


def _load_manifest(path):
    """Manifest format (YAML or JSON):
        models:
          - tag: linear
            config: configs/c_routers/linear.yaml
            checkpoint: results/checkpoints/routers_linear_..._best.pt
          - tag: product_key
            config: configs/c_routers/product_key.yaml
            checkpoint: results/checkpoints/routers_product_key_..._best.pt
    """
    import yaml
    with open(path) as f:
        m = yaml.safe_load(f)
    return [(d["tag"], d["config"], d["checkpoint"]) for d in m["models"]]


def gate_stats(weights):
    """Stats from a (T, top_k) tensor of gate softmax weights."""
    t = weights.float()
    eps = 1e-12
    H = -(t * (t + eps).log()).sum(-1)            # per-token entropy
    log_k = torch.log(torch.tensor(t.shape[-1], dtype=torch.float32))
    Hn = (H / log_k).mean().item()                # normalized entropy
    return {
        "mean":      t.mean().item(),
        "std":       t.std().item(),
        "max_mean":  t.max(-1).values.mean().item(),  # mean of per-token max
        "max_p95":   t.max(-1).values.quantile(0.95).item(),
        "Hnorm":    Hn,                                # 1=uniform, 0=peaked
        "n_tokens":  t.shape[0],
    }


def collect_for_checkpoint(tag, cfg, ckpt):
    print(f"\n>>> [{tag}]  cfg={cfg}", flush=True)
    config = load_config(cfg)
    model, tok = build_model(config)
    load_checkpoint(ckpt, model)
    model = model.cpu().float().eval()

    K = config.get("num_experts", 0)
    top_k = config.get("top_k", 0)
    router_type = config.get("router_type", "linear")
    print(f"    K={K} top_k={top_k} router={router_type}", flush=True)

    module_names = [n for n, m in model.named_modules() if isinstance(m, RoutedLoRA)]
    # per_module[name][dataset] = stats dict
    per_module = {n: {} for n in module_names}

    t0 = time.time()
    for ds in DATASETS:
        for split in ("validation", "test"):
            try:
                inputs, _refs = load_raw_dataset(ds, split=split)
                break
            except Exception:
                continue
        else:
            inputs = []
        if not inputs:
            print(f"  [{ds}] no data", flush=True)
            continue
        n = min(N_SAMPLES, len(inputs))

        # Per-module accumulator: list of (T, top_k) gate-weight tensors
        ds_buf = defaultdict(list)

        for i in range(n):
            enc = tok(inputs[i], return_tensors="pt", truncation=True, max_length=MAX_LEN)
            with torch.no_grad():
                _ = model(**enc)
            for name, m in model.named_modules():
                if isinstance(m, RoutedLoRA) and m._last_topk_weights is not None:
                    w = m._last_topk_weights.reshape(-1, m.top_k)
                    ds_buf[name].append(w)

        for name in module_names:
            if not ds_buf[name]:
                continue
            cat = torch.cat(ds_buf[name], dim=0)
            per_module[name][ds] = gate_stats(cat)

        print(f"  [{ds:14s}] n={n}", flush=True)

    wall = time.time() - t0
    print(f"    collected in {wall:.1f}s", flush=True)

    # Cross-module summary: median of each stat across modules, separately per
    # dataset and pooled across datasets.
    pooled = defaultdict(list)
    for name in module_names:
        for ds, s in per_module[name].items():
            for k, v in s.items():
                pooled[k].append(v)
    pooled_summary = {k: float(torch.tensor(v).median().item()) for k, v in pooled.items()
                      if k != "n_tokens"}

    del model, tok
    gc.collect()

    return {
        "tag": tag,
        "K": K,
        "top_k": top_k,
        "router_type": router_type,
        "checkpoint": ckpt,
        "config": cfg,
        "n_samples_per_dataset": N_SAMPLES,
        "max_len": MAX_LEN,
        "datasets": DATASETS,
        "modules": module_names,
        "per_module": per_module,
        "pooled_summary": pooled_summary,
        "wall_s": wall,
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--manifest", default="results/analysis_manifest.yaml",
                   help="YAML manifest listing checkpoints to analyze.")
    p.add_argument("--output", default="results/analysis/gate_magnitudes.json")
    p.add_argument("--samples", type=int, default=int(os.environ.get("DEMO_SAMPLES", "5")),
                   help="Per-dataset samples to walk.")
    p.add_argument("--max_len", type=int, default=int(os.environ.get("DEMO_MAX_LEN", "128")))
    args = p.parse_args()

    global N_SAMPLES, MAX_LEN
    N_SAMPLES = args.samples
    MAX_LEN = args.max_len

    if not os.path.exists(args.manifest):
        raise FileNotFoundError(
            f"manifest not found: {args.manifest}\n"
            f"Create one by listing your trained checkpoints, e.g.:\n\n"
            f"models:\n"
            f"  - tag: linear\n"
            f"    config: configs/c_routers/linear.yaml\n"
            f"    checkpoint: results/checkpoints/routers_linear_..._best.pt\n"
            f"  - tag: product_key\n"
            f"    config: configs/c_routers/product_key.yaml\n"
            f"    checkpoint: results/checkpoints/routers_product_key_..._best.pt\n"
        )
    models = _load_manifest(args.manifest)

    t_start = time.time()
    out = []
    for tag, cfg, ckpt in models:
        if not os.path.exists(ckpt):
            print(f"\n!!! [{tag}] checkpoint missing: {ckpt}  — skipping")
            continue
        out.append(collect_for_checkpoint(tag, cfg, ckpt))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(f"\nwrote {args.output}  ({os.path.getsize(args.output)/1024:.1f} KB)")
    print(f"[total wall: {time.time()-t_start:.1f}s]")

    # Print a side-by-side table of pooled summaries
    print("\n=== Pooled gate-magnitude summary (median across modules+datasets) ===")
    if out:
        keys = ["mean", "std", "max_mean", "max_p95", "Hnorm"]
        hdr = "router".ljust(18) + "".join(f"{k:>11s}" for k in keys)
        print(hdr)
        for r in out:
            row = r["tag"].ljust(18) + "".join(
                f"{r['pooled_summary'].get(k, float('nan')):>11.4f}" for k in keys
            )
            print(row)


if __name__ == "__main__":
    main()
