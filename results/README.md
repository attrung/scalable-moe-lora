# Results

Eval JSONs and analysis JSONs from the seed-42 run of Parts A-D. Checkpoints are not committed (they are ~130 MB each, gitignored under `results/checkpoints/`); to reproduce them, follow `docs/reproduce.md`.

## Layout

```
results/
├── eval/        # per-checkpoint evaluation outputs (committed, ~25 JSONs)
│   ├── *_eval.json     — in-distribution eval over the 18-dataset training suite
│   └── *_ood_OOD.json  — out-of-distribution eval over the 7-benchmark OOD sweep
├── analysis/    # routing-behavior analysis outputs (committed)
│   ├── k64_<router>.json    — per-module top-k routing data per dataset (Part D)
│   ├── k8_coarse_*.json     — same, for the K=8 coarse pair (best + final ckpts)
│   ├── _summary.json        — cross-checkpoint summary (entropy, dead-exp, Jaccard)
│   └── _gate_magnitudes.json — per-router gate-magnitude statistics (C1 follow-up)
├── tables/      # reserved for derived markdown summary tables
└── checkpoints/ # GITIGNORED — populate by training (or copy yours here)
```

## Reading an eval JSON

Each `*_eval.json` has shape:

```json
{
  "metrics": {
    "<dataset>": {
      "accuracy": <float>,           // for reasoning / MCQ datasets
      "bleu": <float>, "rougeL": <float>  // for NLG datasets (e2e, samsum, mbpp)
    },
    ...
  },
  "routing": {  // present only if --routing_analysis was passed at eval time
    "<module_name>": { "<dataset>": { "<expert_id>": count, ... }, ... }
  }
}
```

The pooled in-distribution accuracy reported in the README and `docs/results.md` is the mean of the `accuracy` field over the 16 accuracy-metric training datasets. The pooled OOD accuracy is the mean over the 6 accuracy-metric OOD benchmarks (IFEval is BLEU/ROUGE-L and is reported separately).

## Reading an analysis JSON

`results/analysis/<tag>.json` from `per_layer_routing.py`:

```json
{
  "tag": "k64_linear",
  "router_type": "linear",
  "K": 64, "top_k": 16,
  "modules": ["model.layers.0.self_attn.q_proj.lora", ..., "model.layers.15.self_attn.v_proj.lora"],
  "data": {
    "<module_name>": {
      "<dataset>": { "counter": {"<expert_id>": count, ...}, "sample_top1s": [<id>, ...] }
    }
  }
}
```

`results/analysis/_gate_magnitudes.json` from `gate_magnitudes.py`:

```json
[
  { "tag": "k64_linear",
    "router_type": "linear",
    "pooled_summary": {"mean": ..., "std": ..., "max_mean": ..., "max_p95": ..., "Hnorm": ...},
    "per_module": {"<module_name>": {"<dataset>": {... gate stats ...}}}
  },
  ...
]
```

Both are consumed by `analysis/per_layer_summary.py`.
