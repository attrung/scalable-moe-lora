"""Utility functions: seeding, config loading, checkpointing, logging."""

import os
import random
import yaml
import torch
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(model, optimizer, epoch, step, val_loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Only save LoRA parameters (trainable ones)
    trainable_state = {k: v for k, v in model.state_dict().items() if "lora" in k.lower()}
    torch.save({
        "model_state_dict": trainable_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "val_loss": val_loss,
    }, path)


def _remap_legacy_router_key(k, model_state):
    """Remap old MoELoRA router key to the current LinearRouter-wrapped path.

    Pre-refactor: MoELoRA.router was a bare nn.Linear → `...lora.router.weight`.
    Post-refactor: MoELoRA.router is a LinearRouter whose .router is the Linear
    → `...lora.router.router.weight`. Returns the remapped key if the new path
    exists in the model, else the original key.
    """
    if k in model_state:
        return k
    if k.endswith(".router.weight"):
        remapped = k[: -len(".router.weight")] + ".router.router.weight"
        if remapped in model_state:
            return remapped
    return k


def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location="cpu")
    # Load only the LoRA parameters
    model_state = model.state_dict()
    loaded_keys = []
    remapped_count = 0
    for k, v in checkpoint["model_state_dict"].items():
        target = _remap_legacy_router_key(k, model_state)
        if target != k:
            remapped_count += 1
        if target in model_state:
            model_state[target] = v
            loaded_keys.append(target)
        else:
            print(f"  WARNING: checkpoint key '{k}' not found in model, skipping")
    if not loaded_keys:
        raise RuntimeError(f"No LoRA weights loaded from {path} — config mismatch?")
    if remapped_count > 0:
        print(f"  Remapped {remapped_count} legacy router key(s) to LinearRouter-wrapped path")
    print(f"  Loaded {len(loaded_keys)} LoRA parameter tensors from checkpoint")
    model.load_state_dict(model_state)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0), checkpoint.get("val_loss", float("inf"))


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen
