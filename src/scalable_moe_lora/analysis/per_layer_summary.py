"""Per-layer routing analysis from collected JSONs in results/analysis/.

Consumes the JSONs written by per_layer_routing.py and computes, per model:

    Per-module metrics (each of 64 MoELoRA modules):
      - marginal expert counts (aggregated across 18 datasets)
      - normalized marginal entropy       H(marginal) / log2(K)
      - hot set                            top-(K/8) experts by marginal mass
      - top-1 mode per dataset              argmax Counter(top-1 per sample)
      - distinct top-1 count                |{top-1 mode per dataset}|

    Per-layer aggregations across the 64 modules:
      - distribution of per-module distinct-top-1 counts (min / median / max)
      - distribution of per-module normalized entropy
      - hot-set Jaccard between layers
      - early (0-10) / middle (11-21) / late (22-31) group averages

Comparisons across models are printed side-by-side.

Run:
    cd <repo-root> && source env.sh >/dev/null
    python3 scripts/per_layer_summary.py                       # all JSONs
    python3 scripts/per_layer_summary.py k8_coarse_best        # subset

Writes a condensed machine-readable summary to
  results/analysis/_summary.json
and prints human-readable tables to stdout.
"""

import sys
import os
import json
import math
import statistics
from collections import Counter


DATA_DIR = "results/analysis"


def load_run(tag_or_path):
    path = tag_or_path if tag_or_path.endswith(".json") else os.path.join(DATA_DIR, f"{tag_or_path}.json")
    with open(path) as f:
        return json.load(f)


def parse_layer(module_name):
    """model.layers.12.self_attn.q_proj.lora -> (12, 'q_proj')."""
    parts = module_name.split(".")
    idx = parts.index("layers")
    layer_i = int(parts[idx + 1])
    # proj name is second-to-last
    proj = parts[-2]  # q_proj or v_proj
    return layer_i, proj


def normalized_entropy(counts, K):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    H = 0.0
    for e in range(K):
        c = counts.get(e, 0)
        if c == 0:
            continue
        p = c / total
        H -= p * math.log2(p)
    H_max = math.log2(K)
    return H / H_max if H_max > 0 else 0.0


def hot_set(marginal_counts, K, hot_n=None):
    """Return the hot_n experts with largest marginal mass (default K/8)."""
    if hot_n is None:
        hot_n = max(1, K // 8)
    total = sum(marginal_counts.values())
    if total == 0:
        return set()
    ranked = sorted(marginal_counts.items(), key=lambda kv: -kv[1])
    return {int(e) for e, _ in ranked[:hot_n]}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def analyze_module(module_data, K, datasets):
    """Returns a dict of per-module metrics for a single MoELoRA module.
    module_data is {dataset: {"counter": {eid: count}, "sample_top1s": [...]}}."""
    marginal = Counter()
    for ds in datasets:
        if ds not in module_data:
            continue
        # JSON keys come back as strings; convert to int.
        for e, c in module_data[ds]["counter"].items():
            marginal[int(e)] += c

    top1_mode_per_ds = {}
    for ds in datasets:
        if ds not in module_data:
            continue
        tops = module_data[ds]["sample_top1s"]
        if tops:
            mode, _ = Counter(tops).most_common(1)[0]
            top1_mode_per_ds[ds] = int(mode)

    distinct_top1 = len(set(top1_mode_per_ds.values()))
    H_norm = normalized_entropy(marginal, K)
    hs = hot_set(marginal, K)
    total = sum(marginal.values())
    hot_capture = (sum(marginal.get(e, 0) for e in hs) / total) if total else 0.0
    dead = sum(1 for e in range(K) if marginal.get(e, 0) == 0)

    # per-dataset top-1 share (how concentrated is each dataset's top-1 signal)
    per_ds_top1_share = {}
    for ds in datasets:
        if ds in module_data:
            tops = module_data[ds]["sample_top1s"]
            if tops:
                mode, count = Counter(tops).most_common(1)[0]
                per_ds_top1_share[ds] = count / len(tops)

    return {
        "marginal": {int(e): int(c) for e, c in marginal.items()},
        "H_norm": H_norm,
        "hot_set": sorted(hs),
        "hot_capture": hot_capture,
        "dead": dead,
        "distinct_top1": distinct_top1,
        "top1_mode_per_ds": top1_mode_per_ds,
        "per_ds_top1_share_mean": statistics.mean(per_ds_top1_share.values()) if per_ds_top1_share else 0.0,
    }


def layer_groups(layer_indices):
    """Split sorted indices into three roughly equal early/mid/late groups.
    Works for LLaMA 3.2 1B (16 layers) and any other layer count."""
    ordered = sorted(layer_indices)
    n = len(ordered)
    if n == 0:
        return [], [], []
    a = n // 3
    b = 2 * n // 3
    return ordered[:a], ordered[a:b], ordered[b:]


def analyze_run(run):
    K = run["K"]
    top_k = run["top_k"]
    router = run["router_type"]
    datasets = run["datasets"]
    module_names = run["modules"]

    per_mod = {}
    for name in module_names:
        if name in run["data"]:
            per_mod[name] = analyze_module(run["data"][name], K, datasets)

    # group keys by (layer_index, proj)
    by_qv = {"q_proj": {}, "v_proj": {}}
    for name, metrics in per_mod.items():
        li, proj = parse_layer(name)
        by_qv[proj][li] = metrics

    # Aggregate scalars across the 64 modules
    all_mods = list(per_mod.values())
    distinct_top1_list = [m["distinct_top1"] for m in all_mods]
    entropy_list = [m["H_norm"] for m in all_mods]
    hot_capture_list = [m["hot_capture"] for m in all_mods]
    dead_list = [m["dead"] for m in all_mods]

    # Pairwise Jaccard of hot sets across all layer pairs (within same proj stream, to avoid mixing q and v)
    pairwise_j = {"q_proj": [], "v_proj": []}
    for proj in ("q_proj", "v_proj"):
        layers_in_order = sorted(by_qv[proj].keys())
        for i, li in enumerate(layers_in_order):
            for lj in layers_in_order[i + 1 :]:
                s1 = set(by_qv[proj][li]["hot_set"])
                s2 = set(by_qv[proj][lj]["hot_set"])
                pairwise_j[proj].append(jaccard(s1, s2))

    # Jaccard of q vs v within each layer
    qv_within_layer_j = []
    layers_both = sorted(set(by_qv["q_proj"]) & set(by_qv["v_proj"]))
    for li in layers_both:
        s1 = set(by_qv["q_proj"][li]["hot_set"])
        s2 = set(by_qv["v_proj"][li]["hot_set"])
        qv_within_layer_j.append(jaccard(s1, s2))

    # Early/mid/late entropy averages (q_proj stream, as representative)
    def _group_mean(proj, selector):
        vals = []
        layers = sorted(by_qv[proj].keys())
        early, mid, late = layer_groups(layers)
        g_early = [selector(by_qv[proj][li]) for li in early]
        g_mid = [selector(by_qv[proj][li]) for li in mid]
        g_late = [selector(by_qv[proj][li]) for li in late]
        return {
            "early": statistics.mean(g_early) if g_early else None,
            "mid": statistics.mean(g_mid) if g_mid else None,
            "late": statistics.mean(g_late) if g_late else None,
        }

    groups_entropy = {
        "q_proj": _group_mean("q_proj", lambda m: m["H_norm"]),
        "v_proj": _group_mean("v_proj", lambda m: m["H_norm"]),
    }
    groups_distinct = {
        "q_proj": _group_mean("q_proj", lambda m: m["distinct_top1"]),
        "v_proj": _group_mean("v_proj", lambda m: m["distinct_top1"]),
    }

    # Worst-layer indicators (most-collapsed)
    worst_layer_min_distinct = min(distinct_top1_list)
    worst_layer_min_entropy = min(entropy_list)
    # For K=8, top_k=2: fully-uniform would have K/8 == 1 hot set and 8 distinct top-1s max;
    # for K=64 top_k=16: uniform would have all 64 used equally, distinct_top1 could be any of 64.

    summary = {
        "tag": run["tag"],
        "label": run["label"],
        "K": K, "top_k": top_k, "router_type": router,
        "n_samples_per_dataset": run.get("n_samples_per_dataset", "?"),
        "n_modules": len(per_mod),
        "distinct_top1": {
            "mean": statistics.mean(distinct_top1_list),
            "median": statistics.median(distinct_top1_list),
            "min": worst_layer_min_distinct,
            "max": max(distinct_top1_list),
            "stdev": statistics.stdev(distinct_top1_list) if len(distinct_top1_list) > 1 else 0.0,
        },
        "entropy_norm": {
            "mean": statistics.mean(entropy_list),
            "median": statistics.median(entropy_list),
            "min": worst_layer_min_entropy,
            "max": max(entropy_list),
        },
        "hot_capture_mean": statistics.mean(hot_capture_list),
        "dead_any_layer": max(dead_list),
        "dead_mean": statistics.mean(dead_list),
        "pairwise_hotset_jaccard_qproj_mean": statistics.mean(pairwise_j["q_proj"]) if pairwise_j["q_proj"] else None,
        "pairwise_hotset_jaccard_vproj_mean": statistics.mean(pairwise_j["v_proj"]) if pairwise_j["v_proj"] else None,
        "qv_within_layer_hotset_jaccard_mean": statistics.mean(qv_within_layer_j) if qv_within_layer_j else None,
        "groups_entropy": groups_entropy,
        "groups_distinct_top1": groups_distinct,
        "per_module": per_mod,
        "by_proj_layer": {proj: {str(li): m for li, m in d.items()} for proj, d in by_qv.items()},
    }
    return summary


def print_summary(s):
    print(f"\n{'='*78}")
    print(f"{s['label']}")
    print(f"{'='*78}")
    print(f"K={s['K']}, top_k={s['top_k']}, router={s['router_type']}, "
          f"n_modules={s['n_modules']}, samples/ds={s['n_samples_per_dataset']}")

    d = s["distinct_top1"]
    print(f"\n  Distinct top-1 modes per module (across 18 datasets):")
    print(f"    min={d['min']}  median={d['median']}  mean={d['mean']:.2f}  max={d['max']}  std={d['stdev']:.2f}")
    e = s["entropy_norm"]
    print(f"  Normalized marginal entropy per module:")
    print(f"    min={e['min']:.3f}  median={e['median']:.3f}  mean={e['mean']:.3f}  max={e['max']:.3f}")
    print(f"  Hot capture (top-K/8 marginal mass) mean: {s['hot_capture_mean']*100:.1f}%")
    print(f"  Dead experts: max over layers = {s['dead_any_layer']},  mean = {s['dead_mean']:.1f}")
    jp = s["pairwise_hotset_jaccard_qproj_mean"]
    jv = s["pairwise_hotset_jaccard_vproj_mean"]
    jqv = s["qv_within_layer_hotset_jaccard_mean"]
    if jp is not None:
        print(f"  Pairwise hot-set Jaccard (all layer pairs):  q_proj={jp:.3f}  v_proj={jv:.3f}")
    if jqv is not None:
        print(f"  Within-layer q vs v hot-set Jaccard:  {jqv:.3f}")

    ge = s["groups_entropy"]["q_proj"]
    gd = s["groups_distinct_top1"]["q_proj"]
    print(f"\n  Early/mid/late layer groups (q_proj stream):")
    def _fmt(v): return "---" if v is None else f"{v:.3f}" if isinstance(v, float) else f"{v:.2f}"
    print(f"    entropy   : early {_fmt(ge['early'])}  mid {_fmt(ge['mid'])}  late {_fmt(ge['late'])}")
    print(f"    distinct  : early {_fmt(gd['early'])}  mid {_fmt(gd['mid'])}  late {_fmt(gd['late'])}")


def print_cross_model_table(summaries):
    print(f"\n{'='*78}")
    print(f"Cross-model table  (per-layer, not aggregated-across-layers)")
    print(f"{'='*78}\n")
    hdr = (
        "tag", "K", "router",
        "distinct_top1 (min/med/max)",
        "H_norm (min/med/max)",
        "dead_max",
        "J_q", "J_v",
    )
    print(f"  {hdr[0]:20s} {hdr[1]:>3s} {hdr[2]:14s} {hdr[3]:>30s}  {hdr[4]:>28s}  {hdr[5]:>8s}  {hdr[6]:>5s} {hdr[7]:>5s}")
    print("  " + "-" * 130)
    for s in summaries:
        d = s["distinct_top1"]; e = s["entropy_norm"]
        jp = s["pairwise_hotset_jaccard_qproj_mean"]
        jv = s["pairwise_hotset_jaccard_vproj_mean"]
        print(f"  {s['tag']:20s} {s['K']:>3d} {s['router_type']:14s} "
              f"{d['min']:>8d} / {d['median']:>4.1f} / {d['max']:<6d}  "
              f"{e['min']:>6.3f} / {e['median']:>5.3f} / {e['max']:<6.3f}  "
              f"{s['dead_any_layer']:>8d}  "
              f"{jp if jp is not None else float('nan'):>5.2f} "
              f"{jv if jv is not None else float('nan'):>5.2f}")


def main():
    tags = sys.argv[1:] if len(sys.argv) > 1 else None
    if tags is None:
        # pick up all *.json except _summary.json
        files = sorted(
            f for f in os.listdir(DATA_DIR)
            if f.endswith(".json") and not f.startswith("_")
        )
        tags = [f[:-5] for f in files]
    if not tags:
        print(f"No JSONs found in {DATA_DIR}; run per_layer_routing.py first.")
        return

    summaries = []
    for tag in tags:
        try:
            run = load_run(tag)
        except FileNotFoundError:
            print(f"!! missing {tag}.json — skipping")
            continue
        s = analyze_run(run)
        summaries.append(s)
        print_summary(s)

    print_cross_model_table(summaries)

    # Write condensed cross-model summary
    condensed = []
    for s in summaries:
        c = {k: v for k, v in s.items() if k not in ("per_module", "by_proj_layer")}
        condensed.append(c)
    out_path = os.path.join(DATA_DIR, "_summary.json")
    with open(out_path, "w") as f:
        json.dump(condensed, f, indent=2)
    print(f"\n[wrote {out_path}]")


if __name__ == "__main__":
    main()
