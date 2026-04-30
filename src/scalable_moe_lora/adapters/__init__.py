"""Adapter zoo: Standard LoRA + MoE-LoRA + TM-LoRA + RoutedLoRA + DispatchMoELoRA.

All adapters wrap an nn.Linear via `LinearWithLoRA`. Three implementations of
the MoE-LoRA architecture (`MoELoRA`, `RoutedLoRA`, `DispatchMoELoRA`) are
numerically equivalent at matched (K, r, top_k); see
`scalable_moe_lora.analysis.correctness`. `RoutedLoRA` is the production path
because it has the smallest activation-memory footprint at large K. Eight
router types are shared across `RoutedLoRA` and `DispatchMoELoRA` (see
`routers.py`).
"""

from .base import LoRA, LinearWithLoRA
from .moe import MoELoRA
from .tm import TMLoRA
from .routed import RoutedLoRA
from .dispatch import (
    DispatchMoELoRA,
    collect_aux_loss,
    collect_distill_loss,
    collect_full_scores,
    set_teacher_scores,
)
from .routers import build_router

__all__ = [
    "LoRA", "LinearWithLoRA",
    "MoELoRA", "TMLoRA", "RoutedLoRA", "DispatchMoELoRA",
    "collect_aux_loss", "collect_distill_loss",
    "collect_full_scores", "set_teacher_scores",
    "build_router",
]
