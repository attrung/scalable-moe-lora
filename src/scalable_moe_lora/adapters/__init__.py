"""Adapter zoo: Standard LoRA + MoE-LoRA + TM-LoRA.

`MoELoRA` is the shared-bottleneck implementation: one `A: (Kr, d)` and one
`B: (d, Kr)`, partitioned into K expert groups of rank r and gated at the
bottleneck. Mathematically identical to the textbook stack-and-gather form
(Luo et al. 2024) and to a sort-by-expert dispatch form, but with the
smallest activation-memory footprint at large K.

Routers are pluggable via `build_router(kind, ...)`; nine kinds are supported
(see `routers.py`).
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
