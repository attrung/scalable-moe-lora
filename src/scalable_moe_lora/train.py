"""Training loop for LoRA fine-tuning of LLaMA (multi-task)."""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.cuda.empty_cache()

import argparse
import sys
import time
import json
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from scalable_moe_lora.utils import set_seed, load_config, save_checkpoint, count_parameters
from scalable_moe_lora.adapters import (
    collect_aux_loss,
    collect_distill_loss,
    collect_full_scores,
    set_teacher_scores,
)
from scalable_moe_lora.utils import load_checkpoint
from scalable_moe_lora.model import build_model
from scalable_moe_lora.data.nlg import load_multitask_dataset, make_collate_fn

RESULTS_DIR = "results"


class DualLogger:
    """Write to both stdout and a log file."""
    def __init__(self, log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(log_path, "w", buffering=1)

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def validate(model, val_loader, device, loss_fn=None):
    import math
    model.eval()
    total_loss = 0.0
    total_steps = 0
    skipped_nonfinite = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            if loss_fn is not None:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                shift_logits = outputs.logits[..., :-1, :].contiguous().float()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss.float()
            loss_val = loss.item()
            if not math.isfinite(loss_val):
                skipped_nonfinite += 1
                continue
            total_loss += loss_val
            total_steps += 1
    if skipped_nonfinite > 0:
        print(f"    [validate] skipped {skipped_nonfinite} non-finite batches")
    if total_steps == 0:
        return float("inf")
    return total_loss / total_steps


def quick_eval(model, tokenizer, device, dataset_name, max_samples=100):
    """Quick BLEU/ROUGE eval on a subset — uses first available dataset."""
    try:
        from scalable_moe_lora.evaluate import generate_predictions, compute_metrics
        from scalable_moe_lora.data.nlg import load_raw_dataset
        inputs, references = load_raw_dataset(dataset_name, split="test")
        inputs = inputs[:max_samples]
        references = references[:max_samples]
        predictions = generate_predictions(model, tokenizer, inputs, device, max_new_tokens=64, beam_size=1)
        metrics = compute_metrics(predictions, references)
        return metrics
    except Exception as e:
        return {"bleu": -1, "rougeL": -1, "error": str(e)}


def train(config, dataset_names, seed, max_steps=None, smoke_with_val=False,
          teacher_config=None, teacher_ckpt=None, distill_coef=0.0,
          resume_from=None):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = config.get("name", "model")
    ds_tag = "+".join(dataset_names)

    log_dir = os.path.join(RESULTS_DIR, "logs")
    checkpoint_dir = os.path.join(RESULTS_DIR, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{run_name}_{ds_tag}_seed{seed}.log")
    logger = DualLogger(log_path)
    sys.stdout = logger

    print(f"\n{'='*70}")
    print(f"  Training: {run_name} | Datasets: {ds_tag} | Seed: {seed}")
    print(f"  Log file: {log_path}")
    print(f"{'='*70}")

    model, tokenizer = build_model(config)
    model = model.to(device)

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"  VRAM reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")

    total_params, trainable_params, frozen_params = count_parameters(model)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Frozen params:    {frozen_params:,}")
    print(f"  Trainable %:      {100 * trainable_params / total_params:.4f}%")

    batch_size = config.get("batch_size", 4)
    max_seq_len = config.get("max_seq_len", 512)
    grad_accum_steps = config.get("gradient_accumulation_steps", 4)

    # VRAM stress test: simulate worst-case batch (all sequences at max_seq_len)
    if torch.cuda.is_available():
        print(f"\n  VRAM stress test (batch={batch_size}, seq={max_seq_len})...")
        try:
            dummy_ids = torch.randint(0, 1000, (batch_size, max_seq_len), device=device)
            dummy_mask = torch.ones_like(dummy_ids)
            dummy_labels = dummy_ids.clone()
            out = model(input_ids=dummy_ids, attention_mask=dummy_mask)
            logits = out.logits[..., :-1, :].contiguous().float()
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), dummy_labels[..., 1:].contiguous().view(-1),
                ignore_index=-100,
            )
            loss.backward()
            peak = torch.cuda.max_memory_allocated() / 1e9
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  >> PASS: peak VRAM={peak:.2f}GB / {total_vram:.1f}GB ({100*peak/total_vram:.0f}%)")
            model.zero_grad()
            del dummy_ids, dummy_mask, dummy_labels, out, logits, loss
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            old_batch = batch_size
            while batch_size > 1:
                batch_size -= 1
                try:
                    dummy_ids = torch.randint(0, 1000, (batch_size, max_seq_len), device=device)
                    out = model(input_ids=dummy_ids, attention_mask=torch.ones_like(dummy_ids))
                    out.logits.sum().backward()
                    model.zero_grad()
                    del dummy_ids, out
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    continue
            grad_accum_steps = max(1, (old_batch * config.get("gradient_accumulation_steps", 4)) // batch_size)
            print(f"  >> OOM at batch={old_batch}! Reduced to batch={batch_size}, grad_accum={grad_accum_steps}")
            print(f"  >> Effective batch: {batch_size * grad_accum_steps}")

    print(f"\n  Loading training data...")
    train_dataset = load_multitask_dataset(dataset_names, tokenizer, max_seq_len, split="train")
    print(f"  Total training examples: {len(train_dataset)}")

    print(f"  Loading validation data...")
    val_dataset = load_multitask_dataset(dataset_names, tokenizer, max_seq_len, split="validation")
    print(f"  Total validation examples: {len(val_dataset)}")

    collate = make_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    label_smoothing = config.get("label_smoothing", 0.0)
    if label_smoothing > 0:
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)
    else:
        loss_fn = None

    lr = config.get("lr", 2e-4)
    weight_decay = config.get("weight_decay", 0.01)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    epochs = config.get("epochs", 5)
    warmup_steps = config.get("warmup_steps", 500)
    aux_loss_coef = config.get("aux_loss_coef", 0.01)

    # ---- Distillation setup (optional) ----
    teacher = None
    if teacher_config and teacher_ckpt and distill_coef > 0:
        from scalable_moe_lora.model import build_model as _build_model
        print(f"\n  Distillation: teacher_config={teacher_config}")
        print(f"                teacher_ckpt={teacher_ckpt}")
        print(f"                distill_coef={distill_coef}")
        t_cfg = teacher_config if isinstance(teacher_config, dict) else __import__(
            "src.utils", fromlist=["load_config"]).load_config(teacher_config)
        teacher, _ = _build_model(t_cfg)
        load_checkpoint(teacher_ckpt, teacher)
        teacher = teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False

    steps_per_epoch = len(train_loader) // grad_accum_steps
    total_training_steps = steps_per_epoch * epochs
    if max_steps is not None:
        total_training_steps = min(total_training_steps, max_steps)

    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps, total_training_steps)

    print(f"  Batch size: {batch_size} x {grad_accum_steps} accum = {batch_size * grad_accum_steps} effective")
    print(f"  Steps/epoch: {steps_per_epoch} | Total steps: {total_training_steps}")
    print(f"  Warmup: {warmup_steps} steps | LR: {lr}")
    print(f"{'='*70}\n")

    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 0

    # Resume from a prior checkpoint if requested. Loads model + optimizer state
    # and advances the LR scheduler to the saved global_step. The outer epoch
    # loop is then started at the next epoch index, so a checkpoint saved at
    # the end of epoch N (zero-indexed) resumes at epoch N+1.
    if resume_from is not None:
        print(f"\n  Resuming from {resume_from}")
        epoch_done, global_step, val_loss_done = load_checkpoint(
            resume_from, model, optimizer
        )
        start_epoch = epoch_done + 1
        best_val_loss = val_loss_done
        for _ in range(global_step):
            scheduler.step()
        print(f"    completed epoch {epoch_done+1}/{epochs} at step {global_step}, "
              f"val_loss={val_loss_done:.4f}")
        print(f"    continuing from epoch {start_epoch+1}/{epochs}\n")

    training_log = {
        "config": config,
        "datasets": dataset_names,
        "seed": seed,
        "trainable_params": trainable_params,
        "steps": [],
        "val_losses": [],
    }

    train_start = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_start = time.time()
        epoch_loss_sum = 0
        epoch_aux_sum = 0.0
        epoch_loss_count = 0
        optimizer.zero_grad()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            bar_format="{l_bar}{bar:30}{r_bar}",
            file=logger.terminal,
        )

        for step_in_epoch, batch in enumerate(pbar):
            if max_steps is not None and global_step >= max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Teacher forward (no grad), capture per-module full_scores, deposit
            # on the matching student modules so the student's MoELoRA.forward
            # can compute KL(student || teacher) on its way through.
            if teacher is not None:
                with torch.no_grad():
                    _ = teacher(input_ids=input_ids, attention_mask=attention_mask)
                t_scores = collect_full_scores(teacher)
                set_teacher_scores(model, t_scores)

            if loss_fn is not None:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # Cast to FP32 before loss — FP16 logits overflow in CrossEntropyLoss
                shift_logits = outputs.logits[..., :-1, :].contiguous().float()
                shift_labels = labels[..., 1:].contiguous()
                loss_raw = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss_raw = outputs.loss.float()
            # Load-balance aux loss (zero for non-dispatch configs).
            aux_loss = collect_aux_loss(model).float()
            loss_total = loss_raw + aux_loss_coef * aux_loss
            if teacher is not None:
                distill_loss = collect_distill_loss(model).float()
                loss_total = loss_total + distill_coef * distill_loss
            loss = loss_total / grad_accum_steps
            loss.backward()
            if teacher is not None:
                # Clear after backward: gradient_checkpointing (use_reentrant=False)
                # re-runs the student forward inside backward, and the KL branch in
                # MoELoRA.forward must produce the same set of saved tensors on
                # both passes. Clearing before backward made the recomputed forward
                # skip the KL branch (saved 64 tensors vs 68 originally) and
                # tripped torch.utils.checkpoint's integrity assertion.
                set_teacher_scores(model, {})
            epoch_loss_sum += loss_raw.item()
            epoch_aux_sum += aux_loss.item()
            epoch_loss_count += 1

            if (step_in_epoch + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                running_loss = epoch_loss_sum / epoch_loss_count
                running_aux = epoch_aux_sum / epoch_loss_count
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - train_start
                vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

                pbar.set_postfix_str(
                    f"loss={running_loss:.4f} aux={running_aux:.3f} lr={current_lr:.2e}"
                )

                if global_step % 10 == 0:
                    torch.cuda.empty_cache()

                if global_step % 100 == 0:
                    log_line = (
                        f"[Step {global_step}/{total_training_steps}] "
                        f"loss={running_loss:.4f} | aux={running_aux:.3f} | "
                        f"lr={current_lr:.2e} | "
                        f"elapsed={int(elapsed)}s | VRAM={vram:.1f}GB"
                    )
                    print(f"\n{log_line}")
                    training_log["steps"].append({
                        "step": global_step,
                        "loss": running_loss,
                        "aux_loss": running_aux,
                        "lr": current_lr,
                        "elapsed": elapsed,
                    })

        pbar.close()

        if max_steps is not None and global_step >= max_steps:
            epoch_time = time.time() - epoch_start
            if smoke_with_val:
                # Run validate() + best-tracking + inline eval before exiting
                # so the pipeline end-to-end is exercised on a short run.
                avg_loss_smoke = epoch_loss_sum / max(epoch_loss_count, 1)
                print(f"\n=== Smoke (with val) done at step {global_step} | "
                      f"avg_loss={avg_loss_smoke:.4f} | "
                      f"time={int(epoch_time)//60}m {int(epoch_time)%60}s ===")
                val_loss = validate(model, val_loader, device, loss_fn)
                training_log["val_losses"].append({
                    "step": global_step,
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                })
                metrics = quick_eval(model, tokenizer, device, dataset_names[0], max_samples=100)
                print(
                    f"\n=== Smoke val_loss={val_loss:.4f} | "
                    f"BLEU={metrics.get('bleu', -1):.4f} | "
                    f"ROUGE-L={metrics.get('rougeL', -1):.4f} ===\n"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model, optimizer, epoch, global_step, val_loss,
                        os.path.join(checkpoint_dir, f"{run_name}_{ds_tag}_seed{seed}_best.pt")
                    )
                    print(f"  >> Saved new best checkpoint (val_loss={val_loss:.4f})\n")
            else:
                print(f"\n=== Smoke test done at step {global_step} | "
                      f"avg_loss={epoch_loss_sum/max(epoch_loss_count,1):.4f} | "
                      f"time={int(epoch_time)//60}m {int(epoch_time)%60}s "
                      f"(validation skipped in smoke mode — full val_loader is "
                      f"~163k examples and would blow the smoke wall) ===\n")
            break

        val_loss = validate(model, val_loader, device, loss_fn)
        training_log["val_losses"].append({
            "step": global_step,
            "epoch": epoch + 1,
            "val_loss": val_loss,
        })

        metrics = quick_eval(model, tokenizer, device, dataset_names[0], max_samples=100)
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss_sum / max(epoch_loss_count, 1)

        summary = (
            f"=== Epoch {epoch+1}/{epochs} complete | "
            f"avg_loss={avg_loss:.4f} | val_loss={val_loss:.4f} | "
            f"BLEU={metrics.get('bleu', -1):.4f} | ROUGE-L={metrics.get('rougeL', -1):.4f} | "
            f"time={int(epoch_time)//60}m {int(epoch_time)%60}s ==="
        )
        print(f"\n{summary}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, global_step, val_loss,
                os.path.join(checkpoint_dir, f"{run_name}_{ds_tag}_seed{seed}_best.pt")
            )
            print(f"  >> Saved new best checkpoint (val_loss={val_loss:.4f})\n")

        model.train()

    elapsed_total = time.time() - train_start
    training_log["training_time_seconds"] = elapsed_total
    training_log["best_val_loss"] = best_val_loss
    training_log["total_steps"] = global_step

    save_checkpoint(
        model, optimizer, epochs, global_step, best_val_loss,
        os.path.join(checkpoint_dir, f"{run_name}_{ds_tag}_seed{seed}_final.pt")
    )

    json_log_file = os.path.join(log_dir, f"{run_name}_{ds_tag}_seed{seed}.json")
    with open(json_log_file, "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE: {run_name} | seed={seed}")
    print(f"  Total time: {int(elapsed_total)//60}m {int(elapsed_total)%60}s")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  JSON log: {json_log_file}")
    print(f"  Text log: {log_path}")
    print(f"{'='*70}\n")

    sys.stdout = logger.terminal
    logger.close()

    return model, tokenizer, training_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--datasets", type=str, required=True,
                        help="Comma-separated: e2e,commongen,webnlg,samsum")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_names = args.datasets.split(",")
    train(config, dataset_names, args.seed, args.max_steps)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        # Ensure the full traceback is printed to stderr (captured by run script)
        traceback.print_exc()
        # Also write to a crash log in case stderr is lost
        crash_log = os.path.join(RESULTS_DIR, "logs", "crash.log")
        os.makedirs(os.path.dirname(crash_log), exist_ok=True)
        with open(crash_log, "a") as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"CRASH at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Args: {sys.argv}\n")
            f.write(traceback.format_exc())
        sys.exit(1)
