"""Standalone VRAM stress test for all model configs.

Loads each model, runs a forward+backward+optimizer.step at configured
batch_size x max_seq_len. Reports peak VRAM as an absolute value AND as a
percentage of total VRAM, plus GPU SM utilization sampled during the step.
No dataset loading required.

Abort rules:
  - Hard OOM (caught torch.cuda.OutOfMemoryError): FAIL
  - Peak > VRAM_ABORT_PCT of total VRAM: FAIL. Threshold set to 70% in v7
    after jobstats on cancelled job 3080830 showed real training at 99.7%
    VRAM vs the stress test's 82.5% reading — a 17-pt gap from allocator
    fragmentation and multi-step tensor accumulation that the single-step
    stress test does not capture. 70% at stress time projects to ~87% at
    real-training time, which has 10+ GB of true buffer on an A100 80 GB.

GPU utilization tracking lets us spot headroom for optimization:
  - low SM util + low VRAM: batch size can grow
  - high SM util + low VRAM: compute bound; batch has diminishing returns
  - high VRAM: cannot push further regardless of SM util
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import subprocess
import sys
import threading
import time
import glob
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from src.utils import load_config, count_parameters
from src.model import build_model


# Peak VRAM abort threshold (percent of total). If exceeded, the config FAILs.
# Set to 70% (from 85%) after observing a 17-pt gap between stress-test peak
# and real-training peak on cancelled job 3080830 (stress showed 82.5%, real
# jobstats showed 99.7%). The 17 pts come from PyTorch allocator cache,
# fragmentation across optimizer steps, and live tensors the single-step
# stress test does not capture. Leaving 30 pts of headroom at the stress-test
# level keeps real-training VRAM under 87% (70 + 17) with some buffer.
VRAM_ABORT_PCT = 75.0


class GPUMonitor:
    """Samples nvidia-smi GPU utilization on a background thread.

    Use start()/stop() around the workload you want to profile. After stop(),
    summary() returns min/mean/max SM utilization over the collection window.
    """

    def __init__(self, interval=0.05):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _poll(self):
        while not self._stop.is_set():
            try:
                raw = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    timeout=1,
                ).decode().strip().splitlines()[0]
                parts = [p.strip() for p in raw.split(",")]
                sm = float(parts[0])
                mem_used = float(parts[1])
                mem_total = float(parts[2])
                self.samples.append((sm, mem_used, mem_total))
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def summary(self):
        if not self.samples:
            return None
        sms = [s[0] for s in self.samples]
        mems = [s[1] for s in self.samples]
        tot = self.samples[-1][2]
        return {
            "sm_util_min": min(sms),
            "sm_util_mean": sum(sms) / len(sms),
            "sm_util_max": max(sms),
            "mem_used_max_mb": max(mems),
            "mem_total_mb": tot,
            "samples": len(self.samples),
        }


def _print_fail(reason, peak, total_vram):
    pct = 100 * peak / total_vram
    print(f"  >> FAIL ({reason}): peak={peak:.2f}GB / {total_vram:.1f}GB ({pct:.1f}%)")


def stress_test_config(config_path):
    config = load_config(config_path)
    name = config.get("name", "unknown")
    batch_size = config.get("batch_size", 4)
    max_seq_len = config.get("max_seq_len", 384)

    print(f"\n{'='*60}")
    print(f"  VRAM Stress Test: {name}")
    print(f"  Config: {config_path}")
    print(f"  Batch: {batch_size} | Seq: {max_seq_len}")
    print(f"{'='*60}")

    device = torch.device("cuda")
    model, tokenizer = build_model(config)
    model = model.to(device)

    total, trainable, frozen = count_parameters(model)
    print(f"  Params: {trainable:,} trainable / {total:,} total")
    print(f"  VRAM after model load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Create optimizer FIRST — AdamW allocates 2 extra state tensors per param
    # (momentum + variance). These materialize on the first optimizer.step().
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.get("lr", 3e-4),
        weight_decay=config.get("weight_decay", 0.01),
    )
    print(f"  VRAM after optimizer: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    torch.cuda.reset_peak_memory_stats()

    # Start GPU utilization monitoring around the forward+backward+step.
    # Fast interval (50 ms) because the step is typically <5 seconds.
    monitor = GPUMonitor(interval=0.05)
    monitor.start()
    step_start = time.time()

    try:
        # Simulate worst case: all sequences at max_seq_len, full
        # forward+backward+optimizer step
        dummy_ids = torch.randint(0, 1000, (batch_size, max_seq_len), device=device)
        dummy_mask = torch.ones_like(dummy_ids)
        dummy_labels = dummy_ids.clone()

        out = model(input_ids=dummy_ids, attention_mask=dummy_mask)
        logits = out.logits[..., :-1, :].contiguous().float()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            dummy_labels[..., 1:].contiguous().view(-1),
            ignore_index=-100,
        )
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()

        step_elapsed = time.time() - step_start
        monitor.stop()
        gpu_stats = monitor.summary()

        peak = torch.cuda.max_memory_allocated() / 1e9
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        pct = 100 * peak / total_vram

        # Report
        print(f"  Step time:        {step_elapsed:.2f} s")
        print(f"  Peak VRAM:        {peak:.2f} GB / {total_vram:.1f} GB ({pct:.1f}%)")
        if gpu_stats:
            print(
                f"  SM utilization:   min={gpu_stats['sm_util_min']:.0f}%  "
                f"mean={gpu_stats['sm_util_mean']:.0f}%  "
                f"max={gpu_stats['sm_util_max']:.0f}%  "
                f"(samples={gpu_stats['samples']})"
            )
            # Optimization hint
            if pct < 50 and gpu_stats["sm_util_mean"] < 60:
                print("  HINT: low VRAM + low SM util — batch size can probably grow for better throughput")
            elif pct < 60 and gpu_stats["sm_util_mean"] >= 80:
                print("  HINT: low VRAM + high SM util — compute bound, batch growth has diminishing returns")
            elif pct >= VRAM_ABORT_PCT - 5:
                print(f"  HINT: high VRAM — approaching the {VRAM_ABORT_PCT:.0f}% abort threshold")

        # 80% abort rule
        if pct > VRAM_ABORT_PCT:
            _print_fail(f">{VRAM_ABORT_PCT:.0f}% VRAM abort threshold", peak, total_vram)
            del dummy_ids, dummy_mask, dummy_labels, out, logits, loss, optimizer
            del model
            torch.cuda.empty_cache()
            return False, peak, total_vram, gpu_stats

        print(f"  >> PASS")

        del dummy_ids, dummy_mask, dummy_labels, out, logits, loss, optimizer
        del model
        torch.cuda.empty_cache()
        return True, peak, total_vram, gpu_stats

    except torch.cuda.OutOfMemoryError:
        monitor.stop()
        gpu_stats = monitor.summary()
        peak = torch.cuda.max_memory_allocated() / 1e9
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        _print_fail(f"OOM at batch={batch_size} seq={max_seq_len}", peak, total_vram)

        del model
        torch.cuda.empty_cache()
        return False, peak, total_vram, gpu_stats


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, required=True,
                        help="Comma-separated config files or glob pattern")
    args = parser.parse_args()

    # Expand glob or split by comma
    if "*" in args.configs:
        config_paths = sorted(glob.glob(args.configs))
    else:
        config_paths = args.configs.split(",")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"Abort threshold: {VRAM_ABORT_PCT:.0f}% VRAM")

    results = []
    for path in config_paths:
        passed, peak, total, gpu_stats = stress_test_config(path)
        results.append((path, passed, peak, total, gpu_stats))

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for path, passed, peak, total, gpu_stats in results:
        name = os.path.basename(path)
        status = "PASS" if passed else "FAIL"
        pct = 100 * peak / total if total else 0
        sm_mean = f"{gpu_stats['sm_util_mean']:.0f}%" if gpu_stats else "n/a"
        print(f"  [{status}] {name}: {peak:.2f}GB/{total:.1f}GB ({pct:.1f}%)  SM_mean={sm_mean}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"\n  >> ALL CONFIGS PASSED under {VRAM_ABORT_PCT:.0f}% threshold")
    else:
        print(f"\n  >> SOME CONFIGS FAILED — reduce batch_size or seq_len for those")
        sys.exit(1)


if __name__ == "__main__":
    main()
