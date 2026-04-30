"""Per-layer routing collection for MoE-LoRA checkpoints (CPU-only).

For each checkpoint listed in the manifest, walks the 18 in-distribution
datasets (N samples each) and records top-k routing indices *per MoELoRA
module* (keyed by its qualified name, e.g. `model.layers.12.self_attn.q_proj.lora`).
Saves one JSON per run to `results/analysis/<run_tag>.json`. Downstream summary
is `per_layer_summary.py`.

Why per-module: LLaMA 3.2 1B has 16 layers × {q_proj, v_proj} = 32 MoELoRA
modules; each has its own router weights and learned specialization. Summing
across them confounds layer-level structure with aggregation artifacts.

Run:
    python -m scalable_moe_lora.analysis.per_layer_routing \\
        --manifest results/analysis_manifest.yaml
"""

import os
import time
import gc
import json
from collections import Counter

import torch

from scalable_moe_lora.utils import load_config, load_checkpoint
from scalable_moe_lora.model import build_model
from scalable_moe_lora.data.reasoning import load_raw_dataset
from scalable_moe_lora.adapters import MoELoRA


DATASETS = [
    "gsm8k", "arc", "commonsenseqa", "piqa", "winogrande", "boolq",
    "hellaswag", "math", "openbookqa", "sciq", "mbpp", "logiqa2",
    "drop", "mmlu_aux", "triviaqa", "anli", "e2e", "samsum",
]

N_SAMPLES = int(os.environ.get("DEMO_SAMPLES", "5"))
MAX_LEN = int(os.environ.get("DEMO_MAX_LEN", "128"))


def _load_manifest(path):
    """YAML manifest with a `models` list of {tag, config, checkpoint} dicts."""
    import yaml
    with open(path) as f:
        m = yaml.safe_load(f)
    return [(d["tag"], d.get("label", d["tag"]), d["config"], d["checkpoint"])
            for d in m["models"]]


def load_model_cpu(config_path, ckpt_path):
    config = load_config(config_path)
    model, tok = build_model(config)
    load_checkpoint(ckpt_path, model)
    model = model.cpu().float().eval()
    K = config.get("num_experts", 0)
    top_k = config.get("top_k", 0)
    router_type = config.get("router_type", "linear")
    return model, tok, K, top_k, router_type


def per_module_usage(model):
    """For the *last* forward pass, return {module_name: Counter(expert_id -> count)}
    aggregated across all tokens in that forward."""
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, MoELoRA) and m._last_routing_indices is not None:
            flat = m._last_routing_indices.flatten().tolist()
            out[name] = Counter(flat)
    return out


def run_model(tag, label, cfg, ckpt, out_dir):
    t_load = time.time()
    print(f"\n>>> [{tag}] Loading — {label}", flush=True)
    print(f"    cfg  : {cfg}")
    print(f"    ckpt : {ckpt}")
    model, tok, K, top_k, router_type = load_model_cpu(cfg, ckpt)
    print(f"    loaded in {time.time()-t_load:.1f}s   (K={K}, top_k={top_k}, router={router_type})", flush=True)

    module_names = [n for n, m in model.named_modules() if isinstance(m, MoELoRA)]
    # data[module_name][dataset] = {"counter": {eid: count}, "sample_top1s": [...]}
    data = {m: {} for m in module_names}
    ds_stats = {}

    t0 = time.time()
    for ds in DATASETS:
        for split in ("validation", "test"):
            try:
                inputs, refs = load_raw_dataset(ds, split=split)
                break
            except Exception:
                continue
        else:
            inputs, refs = [], []
        if not inputs:
            print(f"  [{ds}]  no data", flush=True)
            continue
        n = min(N_SAMPLES, len(inputs))

        # Per-module accumulator for this dataset.
        agg = {m: Counter() for m in module_names}
        per_sample_top1s = {m: [] for m in module_names}

        t_ds = time.time()
        total_tokens = 0
        for i in range(n):
            prompt = inputs[i]
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN)
            with torch.no_grad():
                _ = model(**enc)
            usage = per_module_usage(model)
            for m_name in module_names:
                c = usage.get(m_name)
                if c:
                    agg[m_name].update(c)
                    per_sample_top1s[m_name].append(c.most_common(1)[0][0])
            total_tokens += enc["input_ids"].shape[1]

        dt = time.time() - t_ds
        for m_name in module_names:
            data[m_name][ds] = {
                "counter": {int(k): int(v) for k, v in agg[m_name].items()},
                "sample_top1s": per_sample_top1s[m_name],
            }
        ds_stats[ds] = {"n_samples": n, "avg_tokens": total_tokens / n, "wall_s": dt}
        print(f"  [{ds:14s}] {n} samples, avg {total_tokens/n:.0f} tok, fwd {dt:.1f}s", flush=True)

    total_wall = time.time() - t0
    print(f"    collected in {total_wall:.1f}s", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{tag}.json")
    payload = {
        "tag": tag,
        "label": label,
        "config": cfg,
        "checkpoint": ckpt,
        "K": K,
        "top_k": top_k,
        "router_type": router_type,
        "datasets": DATASETS,
        "modules": module_names,
        "n_samples_per_dataset": N_SAMPLES,
        "max_len": MAX_LEN,
        "dataset_stats": ds_stats,
        "collection_wall_s": total_wall,
        "data": data,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"    wrote {out_path}  ({os.path.getsize(out_path)/1024:.1f} KB)", flush=True)

    del model, tok
    gc.collect()


def main():
    import argparse
    global N_SAMPLES
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--manifest", default="results/analysis_manifest.yaml",
                   help="YAML manifest with `models: [{tag, config, checkpoint}]`.")
    p.add_argument("--out_dir", default="results/analysis",
                   help="Directory to write per-checkpoint routing JSONs.")
    p.add_argument("--samples", type=int, default=N_SAMPLES,
                   help="Per-dataset samples to walk.")
    args = p.parse_args()
    N_SAMPLES = args.samples

    if not os.path.exists(args.manifest):
        raise FileNotFoundError(
            f"manifest not found: {args.manifest}\n"
            f"Create one listing the checkpoints to analyze. Example:\n\n"
            f"models:\n"
            f"  - tag: linear\n"
            f"    label: linear router (Part C reference)\n"
            f"    config: configs/c_routers/linear.yaml\n"
            f"    checkpoint: results/checkpoints/routers_linear_..._best.pt\n"
        )
    models = _load_manifest(args.manifest)

    t_start = time.time()
    for tag, label, cfg, ckpt in models:
        if not os.path.exists(ckpt):
            print(f"\n!!! [{tag}] checkpoint missing: {ckpt}  — skipping")
            continue
        run_model(tag, label, cfg, ckpt, args.out_dir)
    print(f"\n[total wall: {time.time()-t_start:.1f}s]")


if __name__ == "__main__":
    main()
