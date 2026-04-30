"""Scalable MoE-LoRA — controlled factorial study of MoE-LoRA design axes
on a frozen LLaMA 3.2 1B backbone.

Public API:
  build_model(config)        — frozen LLaMA + injected adapter
  load_config(yaml_path)     — load a YAML experiment config
  train(config, datasets, seed, ...)
  evaluate_model(...)        — in-distribution + OOD evaluation

Adapters and routers are exposed via `scalable_moe_lora.adapters`.
"""

from .utils import load_config, save_checkpoint, load_checkpoint, set_seed, count_parameters
from .model import build_model

__all__ = [
    "build_model",
    "load_config", "save_checkpoint", "load_checkpoint",
    "set_seed", "count_parameters",
]
