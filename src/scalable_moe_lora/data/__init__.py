"""Dataset loaders.

`reasoning` covers the full 18-dataset training suite (15 reasoning + 3 NLG)
plus the 7 OOD evaluation benchmarks (MMLU, MMLU-Pro, BBH, IFEval, AGIEval,
GPQA-Diamond, TruthfulQA). It re-uses the tokenization + collation primitives
from `nlg`. Most callers want `reasoning`; `nlg` is exposed for the few
NLG-only paths (legacy evaluation, sample generation).
"""

from .nlg import make_collate_fn, CausalLMDataset
from . import nlg, reasoning

__all__ = ["make_collate_fn", "CausalLMDataset", "nlg", "reasoning"]
