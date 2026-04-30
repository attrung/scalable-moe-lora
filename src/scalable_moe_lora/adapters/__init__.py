"""Adapter zoo: Standard LoRA + MoE-LoRA + TM-LoRA.

`MoELoRA` has K LoRA experts of rank `r` and a router that picks top-k experts
per token. Routers are pluggable via `build_router(kind, ...)`; nine kinds are
supported (see `routers.py`).
"""

from .base import LoRA, LinearWithLoRA
from .tm import TMLoRA
from .moe import (
    MoELoRA,
    collect_aux_loss,
    collect_distill_loss,
    collect_full_scores,
    set_teacher_scores,
)
from .routers import build_router

__all__ = [
    "LoRA", "LinearWithLoRA",
    "TMLoRA", "MoELoRA",
    "collect_aux_loss", "collect_distill_loss",
    "collect_full_scores", "set_teacher_scores",
    "build_router",
]
