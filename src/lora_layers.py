"""LoRA module variants.

Includes:
- Standard LoRA (Hu et al., 2021)
- Non-Linear LoRA (GELU activation between A and B)
- Frozen-Half LoRA (A frozen, only B trained)
- MoE-LoRA (Luo et al., 2024) — K independent expert pairs, linear routing
- TM-LoRA (Token-Modulated LoRA) — shared A/B with additive expert vectors
- LinearWithLoRA — wraps a frozen nn.Linear with an additive LoRA branch

The RoutedLoRA class (unified shared-bottleneck routed LoRA with six router
types) is added in later commits as the study progresses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Baseline + Ablations
# ---------------------------------------------------------------------------

class LoRA(nn.Module):
    """Standard LoRA: linear down-projection A, linear up-projection B."""

    def __init__(self, in_features, out_features, rank=4, alpha=32, dropout=0.0, **kwargs):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.B(self.A(self.dropout(x))) * self.scaling


class NonLinearLoRA(nn.Module):
    """GELU activation between A and B (ablation variant)."""

    def __init__(self, in_features, out_features, rank=4, alpha=32, dropout=0.0, **kwargs):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.B(F.gelu(self.A(self.dropout(x)))) * self.scaling


class FrozenHalfLoRA(nn.Module):
    """A is frozen (random projection); only B is trained (ablation variant)."""

    def __init__(self, in_features, out_features, rank=4, alpha=32, dropout=0.0, **kwargs):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)
        self.A.weight.requires_grad = False

    def forward(self, x):
        return self.B(self.A(self.dropout(x))) * self.scaling


# ---------------------------------------------------------------------------
# MoE-style LoRA variants
# ---------------------------------------------------------------------------

class MoELoRA(nn.Module):
    """MoE-LoRA: K independent LoRA experts with top-k linear routing.

    Reference: Luo et al. (2024). MoELoRA: Contrastive Learning Guided
    Mixture of Experts on PEFT for LLMs. arXiv:2402.12851.

    Each expert is an independent (A_i, B_i) pair. A router scores experts,
    selects top-k, and computes a weighted sum of their outputs.

    Note: this loop implementation materializes a (batch, seq, K, out_features)
    stack tensor whose memory scales with K. For K >= 16 it becomes the memory
    bottleneck. The RoutedLoRA class (added in later commits) uses a shared
    bottleneck whose memory cost is constant in K*r.
    """

    def __init__(self, in_features, out_features, rank=4, alpha=32, dropout=0.0,
                 num_experts=8, top_k=2, **kwargs):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.experts_A = nn.ModuleList([
            nn.Linear(in_features, rank, bias=False) for _ in range(num_experts)
        ])
        self.experts_B = nn.ModuleList([
            nn.Linear(rank, out_features, bias=False) for _ in range(num_experts)
        ])
        self.router = nn.Linear(in_features, num_experts, bias=False)

        for A in self.experts_A:
            nn.init.kaiming_uniform_(A.weight)
        for B in self.experts_B:
            nn.init.zeros_(B.weight)

    def forward(self, x):
        x_dropped = self.dropout(x)

        scores = self.router(x)
        top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=-1)
        self._last_routing_indices = top_k_indices.detach()
        top_k_weights = F.softmax(top_k_scores, dim=-1)

        expert_outputs = torch.stack(
            [B(A(x_dropped)) for A, B in zip(self.experts_A, self.experts_B)],
            dim=-2,
        )

        idx = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, expert_outputs.shape[-1])
        selected = torch.gather(expert_outputs, dim=-2, index=idx)
        output = (selected * top_k_weights.unsqueeze(-1)).sum(dim=-2)
        return output * self.scaling


class TMLoRA(nn.Module):
    """Token-Modulated LoRA: shared A/B with additive expert vector modulation.

    Forward: output = GELU(x.A + E_token) . B, where E_token is a top-k
    weighted sum of learned expert vectors.
    """

    def __init__(self, in_features, out_features, rank=4, alpha=32, dropout=0.0,
                 num_experts=8, top_k=4, **kwargs):
        super().__init__()
        self.scaling = alpha / rank
        self.top_k = min(top_k, num_experts)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.expert_vectors = nn.Parameter(torch.randn(num_experts, rank) * 0.01)
        self.router = nn.Linear(in_features, num_experts, bias=False)

        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        x_dropped = self.dropout(x)

        scores = self.router(x)
        top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_scores, dim=-1)

        selected = self.expert_vectors[top_k_indices]
        E_token = (selected * top_k_weights.unsqueeze(-1)).sum(dim=-2)

        hidden = self.A(x_dropped) + E_token
        return self.B(F.gelu(hidden)) * self.scaling


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class LinearWithLoRA(nn.Module):
    """Wraps a frozen nn.Linear with an additive LoRA branch.

    Computes: output = original(x) + lora(x.float()).to(base_dtype)

    The LoRA branch is run in FP32 for training stability while the base
    layer stays in the model's native precision (typically FP16).
    """

    def __init__(self, original_linear, lora_module):
        super().__init__()
        self.original_linear = original_linear
        self.lora = lora_module

    def forward(self, x):
        base_out = self.original_linear(x)
        lora_out = self.lora(x.float()).to(base_out.dtype)
        return base_out + lora_out
