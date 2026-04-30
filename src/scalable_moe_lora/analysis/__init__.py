"""Routing analysis utilities (CPU-only):

  per_layer_routing  — collect per-module top-k expert indices across datasets
  per_layer_summary  — entropy / dead-expert / hot-set Jaccard summaries
  gate_magnitudes    — gate-magnitude (sharpness) statistics per router
  correctness        — numerical-equivalence test for the three MoE-LoRA impls
"""
