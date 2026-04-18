# Future work

This document tracks planned experiments and research directions that follow from the Phase B + C + D results.

## Phase D tail (in progress)

- **`granularity_r1_k64_early_shared` training** — currently running, ~7 hours of wall-clock remaining. Tests whether one routing decision, computed from the first LoRA layer's input and reused at all subsequent LoRA injection points, matches per-layer routing quality. Motivated by the Omni-Router Transformer finding that in deep residual networks the hidden state changes only incrementally between layers. If early-shared matches linear quality, single-decision routing provides a 16-32x amortization of router parameters across LoRA layers, making the router cost practically negligible.
- **`granularity_r1_k64_cosine` OOD eval** — training is complete, OOD eval is queued.

## Phase E — Multi-seed error bars (next after Phase D)

Single-seed results establish a trend but cannot distinguish a real effect from seed-specific noise. Before committing compute to 3B scale-up (Phase F) or to a full-MoE pretraining follow-up, we want statistical confidence in the Phase B + D picture.

- Re-run the **winning Phase D router config + baseline** at seeds 0 and 123. Between 2 and 4 additional training runs depending on whether Phase D reveals a single clear winner or a close tie.
- Extend the 18-dataset in-distribution eval and the 7-benchmark OOD sweep to the new checkpoints.
- Aggregate into per-config mean ± σ across 3 seeds.

**Done when**: mean ± σ is reported for the winning config and the baseline on both in-dist and OOD, with either a clear > 1σ lead for fine-grained routing or a clear null result.

**Also in Phase E (opportunistic)**: re-run `granularity_r8_k8_linear` (which diverged at seed 42) with a load-balancing loss term (Shazeer auxiliary loss or DeepSeek's bias-based balancing). Determines whether the epoch-2 training divergence is a reproducible routing-collapse failure mode specific to coarse granularity + linear routing at top_k=2, or a one-off seed effect. Cheap (single run) and paper-relevant.

## Phase F — 3B scale-up

Replicate the winning router (selected from Phase D, confirmed by Phase E) at LLaMA 3.2 3B:

- Configs to create: `reasoning_3b_baseline`, `reasoning_3b_<winning_router>`. Both need fresh VRAM stress tests under mandatory gradient checkpointing.
- Compute: ~2 days wall (2 training runs at 2-concurrent, each longer than 1B due to model size, + 2 OOD evals).
- Purpose: confirm that both the granularity finding (fine-grained routing helps) and the router finding (product-key matches linear at √K cost) persist at 3x model scale. This is the final evidence that the LoRA results are a reliable signal for full-scale MoE design decisions.

## Phase G — Paper polish + final figures

Compute component:
- **Beam-search NLG eval** on E2E and SAMSum checkpoints (greedy decoding is the current default; beam search typically adds a few BLEU points and is cheap).
- **BBH re-evaluation** on all surviving 1B + 3B checkpoints with the corrected extractor (the current extractor does not strip parenthesized letter targets like `(A)` in ~16 of 26 BBH subtasks).
- **Routing entropy heatmaps** per dataset per granularity for the Phase B routed configs. The routing analysis is already captured in the eval JSONs; this is a plotting pass.

Paper deliverables:
- Granularity-vs-accuracy curves (one per router type) at 1B, with multi-seed error bars from Phase E
- 1B-vs-3B granularity comparison plot from Phase F (the headline figure)
- Router comparison table at K=64: linear, lowrank, hierarchical, product-key, cosine, early-shared — with param counts, in-distribution accuracy, and OOD accuracy, at 1B and 3B
- Analytical router-scaling plot: `O(d·K)` linear vs `O(d·rdim + K·rdim)` lowrank vs `O(d·√K)` hierarchical / product-key, as K grows, at d in {2048, 4096, 8192}. Computed from formulas, not empirically.
- Generalization table (Phase C + Phase F OOD numbers)
- Methods, setup, results, discussion, related work, poster

## Extension directions beyond this paper

### Bringing the findings to full MoE pretraining

The LoRA setting we study isolates the routing contribution: the base model is frozen, so any quality difference between router types comes purely from routing. The 1B + 3B LoRA pipeline acts as a cheap controlled dry-run — if the multi-seed Phase E and scale-up Phase F confirm the granularity and routing findings, that is the trigger to commit full-pretraining compute to the winning design.

- **Product-key routing in MoE pretraining**: integrate product-key scoring into an MoE FFN (e.g., replace the linear router in a DeepSeek-MoE-style architecture). The √K cost scaling should unlock much higher expert counts without inflating router parameters. At DeepSeek-V3's hidden dim d=7168 with K=256, replacing linear routing with product-key routing drops the per-layer router cost from 1.84M parameters to ~114K (16× reduction); at K=4096 the gap widens to 23×.
- **Early-shared routing in pretraining**: if the PEFT result shows early-shared matches per-layer routing, this is even more striking in pretraining where the base model is not frozen and hidden states change more substantially between layers. If it still works, shared routing becomes a major engineering simplification (one routing decision per token per forward pass, regardless of model depth).

### Router-design axes not yet explored

- **Mixture of routers**: multiple small routers whose outputs are combined, either by averaging or by learned gating. Tests whether ensembled routing beats a single large router at matched total parameters.
- **Learned-hashing routers**: each expert is assigned a fixed random hash bucket at init, with a small learned correction term. Tests the quality floor of "nearly free" routing.
- **ReXMoE-style cross-layer expert sharing**: routers at each layer can draw experts from adjacent layers' pools, giving combinatorial diversity without inflating total parameters.
- **Token-level vs sequence-level vs segment-level routing**: whether routing decisions should be made per-token (current), per-sequence (one decision per example), or at some intermediate granularity. Cheaper routing at coarser granularities, with a corresponding quality tradeoff.

### Analysis directions using existing Phase B+C+D data

- **Routing geometry as a learned task taxonomy**: for each of the 18 training datasets, compute a per-dataset routing profile (normalized histogram over K experts, averaged across tokens and layers). Project into 2D via UMAP or t-SNE. If the map shows interpretable task clusters (reasoning vs NLG vs code) at fine granularity but not at coarse, that is evidence that fine-grained routing implicitly learns task structure.
- **Granularity-dependent generalization gap**: plot `mean_indist - mean_OOD` against granularity for each router type. Whether the gap shrinks or widens at finer granularity tells us whether fine-grained routing helps generalization (good) or overfits (bad).
- **Per-layer routing entropy**: quantify how "confident" the router is at each layer. Do some layers route nearly uniformly (collapsed routing) while others show sharp expert selection? This would motivate adaptive per-layer granularity.
