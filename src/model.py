"""LLaMA + LoRA injection logic.

Loads a frozen base model in FP16 to halve memory and bandwidth.
LoRA weights are created in FP32 for training stability.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.lora_layers import (
    LoRA, NonLinearLoRA, FrozenHalfLoRA,
    MoELoRA, TMLoRA, RoutedLoRA,
    LinearWithLoRA,
)


LORA_CLASS_MAP = {
    "standard": LoRA,
    "nonlinear": NonLinearLoRA,
    "frozen_half": FrozenHalfLoRA,
    "moe": MoELoRA,
    "tm": TMLoRA,
    "routed": RoutedLoRA,
}

TARGET_MODULES_MAP = {
    "qv": ["q_proj", "v_proj"],
    "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def get_model_and_tokenizer(model_name="meta-llama/Llama-3.2-1B"):
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def freeze_all_parameters(model):
    for param in model.parameters():
        param.requires_grad = False


def _get_lora_kwargs(config):
    lora_type = config.get("lora_type", "standard")
    kwargs = {}
    if lora_type == "moe":
        kwargs["num_experts"] = config.get("num_experts", 8)
        kwargs["top_k"] = config.get("top_k", 2)
    elif lora_type == "tm":
        kwargs["num_experts"] = config.get("num_experts", 8)
        kwargs["top_k"] = config.get("top_k", 4)
    elif lora_type == "routed":
        kwargs["num_experts"] = config.get("num_experts", 64)
        kwargs["top_k"] = config.get("top_k", 16)
        kwargs["router_type"] = config.get("router_type", "lowrank")
        kwargs["router_dim"] = config.get("router_dim", 16)
    return kwargs


def inject_lora(model, config):
    """Inject LoRA adapters into the target attention/MLP projections.

    LoRA weights are created in FP32 for training stability, wrapped around
    the model's native-precision (FP16) frozen linear layers.
    """
    lora_type = config.get("lora_type", "standard")
    lora_cls = LORA_CLASS_MAP[lora_type]
    rank = config.get("rank", 4)
    alpha = config.get("alpha", 32)
    dropout = config.get("lora_dropout", 0.0)
    target_modules = config.get("target_modules", "qv")
    extra_kwargs = _get_lora_kwargs(config)

    target_names = TARGET_MODULES_MAP[target_modules]

    for layer in model.model.layers:
        for name in target_names:
            if name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                parent = layer.self_attn
            elif name in ("gate_proj", "up_proj", "down_proj"):
                parent = layer.mlp
            else:
                continue

            original = getattr(parent, name)
            in_f = original.in_features
            out_f = original.out_features

            lora = lora_cls(in_f, out_f, rank=rank, alpha=alpha,
                            dropout=dropout, **extra_kwargs)
            wrapped = LinearWithLoRA(original, lora)
            setattr(parent, name, wrapped)

    return model


def build_model(config):
    model_name = config.get("model_name", "meta-llama/Llama-3.2-1B")
    model, tokenizer = get_model_and_tokenizer(model_name)
    freeze_all_parameters(model)
    inject_lora(model, config)

    if config.get("gradient_checkpointing", False):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model, tokenizer
