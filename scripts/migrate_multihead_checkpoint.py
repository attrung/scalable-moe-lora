"""One-shot: migrate a multihead-PK checkpoint from the legacy ModuleList
layout (scorers1.{0..H-1}.weight) to the vectorized layout (scorer1.weight)
that MultiHeadProductKeyRouter expects after vectorization.

The vectorized router replaces H separate Linear(d, sqrt_K) modules with one
Linear(d, H*sqrt_K) — bit-exact equivalent to the loop, but one kernel launch
per scorer instead of H. Existing checkpoints saved with the loop layout can
be migrated by cat-along-dim-0 of the H per-head weight tensors. The same cat
is applied to the optimizer's exp_avg / exp_avg_sq Adam moments so resume
preserves training state exactly.

Usage:
    python scripts/migrate_multihead_checkpoint.py \\
        --config configs/extensions/multihead_pk.yaml \\
        --checkpoint <path-to-old-ckpt.pt> \\
        --output <path-to-migrated.pt>
"""

import argparse
import os
import re

import torch

from scalable_moe_lora.model import build_model
from scalable_moe_lora.utils import load_config


_RE_OLD = re.compile(r"(.+\.router)\.scorers([12])\.(\d+)\.weight$")


def detect_groups(state_keys, num_heads):
    """Group old per-head scorer keys into {new_key: [old_keys in head order]}."""
    raw = {}
    for k in state_keys:
        m = _RE_OLD.match(k)
        if not m:
            continue
        base, scorer_num, head_idx = m.group(1), m.group(2), int(m.group(3))
        new_key = f"{base}.scorer{scorer_num}.weight"
        raw.setdefault(new_key, {})[head_idx] = k
    out = {}
    for nk, heads in raw.items():
        assert sorted(heads.keys()) == list(range(num_heads)), (
            f"incomplete head set for {nk}: got {sorted(heads.keys())}, "
            f"expected 0..{num_heads-1}"
        )
        out[nk] = [heads[h] for h in range(num_heads)]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num_heads", type=int, default=4)
    args = p.parse_args()

    print(f"Loading {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    old_msd = ckpt["model_state_dict"]
    old_osd = ckpt["optimizer_state_dict"]
    print(f"  old model_state_dict: {len(old_msd)} tensors")
    print(f"  old optimizer state:  {len(old_osd['state'])} entries")

    groups = detect_groups(list(old_msd.keys()), args.num_heads)
    print(f"  detected {len(groups)} multihead scorer groups (each {args.num_heads}-way)")
    if not groups:
        print("  No legacy multihead keys found — nothing to migrate.")
        return

    config = load_config(args.config)
    new_model, _ = build_model(config)
    new_qnames = [n for n, p in new_model.named_parameters() if p.requires_grad]
    print(f"  new model has {len(new_qnames)} trainable params")

    old_qnames = list(old_msd.keys())
    old_qname_to_id = {q: i for i, q in enumerate(old_qnames)}

    new_msd = {}
    for nq in new_qnames:
        if nq in groups:
            new_msd[nq] = torch.cat([old_msd[k] for k in groups[nq]], dim=0)
        elif nq in old_msd:
            new_msd[nq] = old_msd[nq]
        else:
            print(f"  WARNING: new param {nq} has no old counterpart")

    new_state = {}
    saved_state = old_osd["state"]
    for nid, nq in enumerate(new_qnames):
        if nq in groups:
            old_states = [saved_state[old_qname_to_id[k]] for k in groups[nq]]
            new_state[nid] = {
                "step": old_states[0]["step"],
                "exp_avg":    torch.cat([s["exp_avg"]    for s in old_states], dim=0),
                "exp_avg_sq": torch.cat([s["exp_avg_sq"] for s in old_states], dim=0),
            }
        elif nq in old_qname_to_id:
            new_state[nid] = saved_state[old_qname_to_id[nq]]

    new_pg = [{**pg, "params": list(range(len(new_qnames)))} for pg in old_osd["param_groups"]]
    new_osd = {"state": new_state, "param_groups": new_pg}

    new_ckpt = {
        "model_state_dict": new_msd,
        "optimizer_state_dict": new_osd,
        "epoch": ckpt["epoch"],
        "step": ckpt["step"],
        "val_loss": ckpt["val_loss"],
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(new_ckpt, args.output)
    print(f"Saved {args.output}  ({os.path.getsize(args.output)/1e6:.1f} MB)")

    print("Verifying migrated checkpoint loads into the new model...")
    merged = new_model.state_dict()
    merged.update(new_msd)
    new_model.load_state_dict(merged)
    print(f"  loaded {len(new_msd)} trainable tensors, no errors")
    print(f"\nResume metadata: epoch={ckpt['epoch']+1} done (zero-indexed: {ckpt['epoch']}), "
          f"step={ckpt['step']}, val_loss={ckpt['val_loss']:.4f}")


if __name__ == "__main__":
    main()
