#!/usr/bin/env python3
"""LoRA healing/capability run for the trimmed Rineya core, built for bit-exact resume.

Design commitments, in priority order:

  1. **Bit-exact resume.** The optimizer sees a plan that is a pure function of
     (mix, seed, epoch, budgets). Resuming restores every RNG stream, the
     optimizer state, and the exact position in that plan, then replays it. A run
     interrupted at step k and resumed produces the same weights as one that ran
     straight through. `scripts/verify/verify_resume.py` proves this rather than
     asserting it.
  2. **No OOM crashes.** A hard allocator cap plus a pre-flight probe at the
     plan's worst-case shape. A budget that cannot fit fails in seconds, before
     step 0, with the number to use instead.
  3. **Clean interruption.** SIGINT/SIGTERM finish the current optimizer step,
     checkpoint atomically, and exit 0. Ctrl-C is a supported way to stop.
  4. **Loud refusal over silent divergence.** Plan-hash, mix-hash and environment
     drift all abort rather than quietly training a different model.

Speed is explicitly traded away for (1): deterministic kernels and the math SDPA
backend are slower than the defaults. That is the intended bargain.

Usage:
  python heal.py --config ../../configs/heal_b0.yaml
  python heal.py --config ../../configs/heal_b0.yaml --dry-run
  python heal.py --config ../../configs/heal_b0.yaml --resume auto
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import checkpoint as ckpt          # noqa: E402
import dataset as ds               # noqa: E402
import determinism                 # noqa: E402
import memory as mem               # noqa: E402
from plan import build_plan        # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
_STOP = {"requested": False, "signal": None}


def _request_stop(signum, _frame):
    if _STOP["requested"]:
        print("\nsecond signal received; exiting immediately without checkpoint")
        raise SystemExit(130)
    _STOP["requested"] = True
    _STOP["signal"] = signum
    print(f"\nsignal {signum} received: finishing the current optimizer step, "
          f"then checkpointing and exiting cleanly. Signal again to abort now.")


def lr_at(step: int, *, base_lr: float, total: int, warmup: int,
          schedule: str, min_ratio: float) -> float:
    """Pure function of step: nothing to checkpoint, nothing to drift."""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    if schedule == "constant":
        return base_lr
    progress = 0.0 if total <= warmup else (step - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    if schedule == "linear":
        factor = 1.0 - progress
    else:  # cosine
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * factor)


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(path)
    return cfg


def build_model(cfg: dict, device: torch.device):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = ROOT / cfg["model"]["path"]
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[cfg["model"].get("dtype", "bfloat16")]
    if device.type == "cpu" and dtype is not torch.float32:
        print(f"note: forcing float32 on CPU (config asked for {dtype})")
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=dtype,
        attn_implementation=cfg["model"].get("attn_implementation", "eager"))
    model.config.use_cache = False

    lora = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora.get("dropout", 0.0),
        bias="none", task_type="CAUSAL_LM",
        target_modules=list(lora["target_modules"])))

    if cfg["train"].get("gradient_checkpointing", True):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    # LoRA parameters in fp32 regardless of the base dtype: Adam moments on bf16
    # parameters lose too much precision to be worth the memory saved.
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()
    model.to(device)
    return tokenizer, model


def forward_loss(model, input_ids, attention_mask, labels, *, normaliser: float,
                 autocast_dtype=None, device_type="cuda"):
    """Summed token loss / normaliser, so accumulation yields a true token mean."""
    context = (torch.autocast(device_type=device_type, dtype=autocast_dtype)
               if autocast_dtype is not None else torch.enable_grad())
    with context:
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    loss_sum = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1), ignore_index=ds.IGNORE_INDEX, reduction="sum")
    counted = int((shift_labels != ds.IGNORE_INDEX).sum().item())
    return loss_sum / normaliser, loss_sum.detach(), counted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default="auto", help="auto | none | <checkpoint path>")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the plan, probe memory, report, and exit")
    ap.add_argument("--max-steps", type=int, default=None, help="smoke-test cap")
    ap.add_argument("--out", default=None, help="override run directory")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    seed = int(cfg["seed"])
    strict = bool(cfg.get("determinism", {}).get("strict", True))
    determinism.enable(seed, strict=strict)

    use_cuda = torch.cuda.is_available() and cfg.get("device", "auto") != "cpu"
    device = torch.device("cuda:0" if use_cuda else "cpu")
    run_dir = Path(args.out) if args.out else ROOT / cfg["run"]["dir"]
    run_dir.mkdir(parents=True, exist_ok=True)

    cap = mem.apply_cap(cfg["memory"].get("cap_mib") if use_cuda else None)
    print(f"device={device}  cap={f'{cap:,.0f} MiB' if cap else 'n/a'}  "
          f"free={mem.free_mib():,.0f} MiB" if use_cuda else f"device={device}")

    tokenizer, model = build_model(cfg, device)
    mix_path = ROOT / cfg["data"]["mix"]
    data = ds.build(mix_path, tokenizer, max_len=int(cfg["data"]["max_len"]),
                    cache_dir=run_dir / "cache",
                    tokenizer_dir=ROOT / cfg["model"]["path"])

    total_lengths, supervised = data.total_lengths, data.supervised_lengths
    epochs = int(cfg["train"]["epochs"])
    plans = [build_plan(
        total_lengths, supervised, seed=seed, epoch=e,
        max_tokens_per_microbatch=int(cfg["memory"]["max_tokens_per_microbatch"]),
        max_rows_per_microbatch=int(cfg["memory"]["max_rows_per_microbatch"]),
        micro_batches_per_step=int(cfg["train"]["grad_accum"]),
        megabatch_rows=int(cfg["data"].get("megabatch_rows", 2048)),
        max_len=int(cfg["data"]["max_len"]),
        width_bucket=int(cfg["data"].get("width_bucket", 1))) for e in range(epochs)]
    total_steps = sum(len(p.steps) for p in plans)
    plan_hash = ";".join(p.hash() for p in plans)

    print(f"\nplan: {epochs} epoch(s), {total_steps:,d} optimizer steps")
    for p in plans:
        s = p.summary()
        pad_eff = s["actual_tokens"] / max(s["padded_positions"], 1)
        sup_frac = s["supervised_tokens"] / max(s["actual_tokens"], 1)
        print(f"  epoch {s['epoch']}: {s['steps']:,d} steps, {s['rows']:,d} rows, "
              f"{s['padded_positions'] / 1e6:.2f}M padded positions")
        print(f"    padding efficiency {pad_eff:.1%} (real tokens / padded), "
              f"supervised {sup_frac:.1%} of real tokens")
        print(f"    widest micro-batch {s['widest_micro_batch']}")

    # Probe a synthetic worst-case step rather than any real one, so the result is
    # an upper bound instead of a sample. Peak memory has two drivers: the widest
    # single micro-batch (activation peak) and the number of distinct widths the
    # allocator has to keep blocks for. No real step maximises both -- the heaviest
    # step by total cost need not contain the widest micro-batch anywhere in the
    # plan. So the probe step is the `grad_accum` most expensive *distinct* shapes
    # found across every epoch, which bounds both drivers at once.
    by_shape: dict[tuple[int, int], int] = {}
    for p in plans:
        for step in p.steps:
            for m in step.micro:
                by_shape[(len(m.indices), m.max_len)] = m.padded_cost
    accum = int(cfg["train"]["grad_accum"])
    probe_shapes = sorted(by_shape, key=lambda k: -by_shape[k])[:accum]
    # Pad out with the widest shape if the plan has fewer distinct shapes than a
    # step has micro-batches, so the probe still represents a full step.
    while len(probe_shapes) < accum:
        probe_shapes.append(probe_shapes[0])
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["train"]["lr"]), betas=(0.9, 0.999),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)), eps=1e-8)

    autocast_dtype = (torch.bfloat16 if use_cuda
                      and cfg["model"].get("dtype") == "bfloat16" else None)

    # The probe must measure the configuration training will actually use.
    # Gradient checkpointing is a no-op in eval mode, so probing before
    # model.train() measures the un-checkpointed forward and over-estimates peak
    # memory badly enough to refuse budgets that would in fact have fit.
    model.train()

    def probe_step(batches):
        normaliser = float(sum(r * s for r, s in probe_shapes))
        # Twice, deliberately, with no cache flush between. A cold allocator
        # under-reports steady state badly -- it has no cached blocks yet, so a
        # single pass measured 2,748 MiB where real training settles at 3,882.
        # The second pass runs against the first pass's cached blocks, which is
        # the regime training actually lives in.
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            for ii, am, lb in batches:
                loss, _, _ = forward_loss(model, ii, am, lb, normaliser=normaliser,
                                          autocast_dtype=autocast_dtype,
                                          device_type=device.type)
                loss.backward()
            # The optimizer step is part of the probe on purpose: Adam's exp_avg
            # and exp_avg_sq are created lazily on first use, so a
            # forward/backward-only probe never accounts for them.
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # The probe mutates weights and creates optimizer state, so both are undone
    # exactly afterwards. Combined with the re-seed below, it cannot influence
    # training -- which verify_determinism.py checks rather than assumes.
    before = {n: p.detach().clone() for n, p in model.named_parameters()
              if p.requires_grad}
    report = mem.probe(probe_step, shapes=probe_shapes, cap_mib=cap,
                       margin_mib=float(cfg["memory"].get("margin_mib", 1024)))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.copy_(before[name])
    del before
    optimizer.state.clear()
    optimizer.zero_grad(set_to_none=True)
    print("\n" + report.render())
    determinism.seed_everything(seed)

    if not report.ok:
        raise SystemExit(
            "\nrefusing to start: the widest micro-batch does not fit the cap.\n"
            f"Set memory.max_tokens_per_microbatch to {report.suggestion:,d} "
            "(or raise memory.cap_mib) and re-run.")

    position = ckpt.Position(epoch=0, step_in_epoch=0, global_step=0,
                             rows_seen=0, supervised_tokens_seen=0)
    if args.resume != "none":
        path = (ckpt.latest_path(run_dir / "checkpoints") if args.resume == "auto"
                else Path(args.resume))
        if path and path.exists():
            payload = ckpt.load(
                path, expect_plan_hash=plan_hash, expect_mix_sha256=data.mix_sha256,
                allow_env_drift=bool(cfg.get("resume", {}).get("allow_env_drift", False)),
                allow_code_drift=bool(cfg.get("resume", {}).get("allow_code_drift", False)))
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(model, payload["adapter_state"])
            optimizer.load_state_dict(payload["optimizer_state"])
            payload["rng_state"].restore()
            position = payload["position_obj"]
            print(f"\nresumed from {path.name}: epoch {position.epoch}, "
                  f"step_in_epoch {position.step_in_epoch}, "
                  f"global_step {position.global_step:,d}")
        else:
            print("\nno checkpoint found; starting fresh")

    if args.dry_run:
        print("\ndry run complete; nothing trained")
        return

    # SIGBREAK is the only signal Windows can reliably deliver to a child process
    # group, so it is registered alongside the POSIX pair rather than assumed away.
    for signame in ("SIGINT", "SIGTERM", "SIGBREAK"):
        handler = getattr(signal, signame, None)
        if handler is not None:
            try:
                signal.signal(handler, _request_stop)
            except (OSError, ValueError):
                print(f"note: could not install a {signame} handler")

    log_path = run_dir / "train_log.jsonl"
    save_every = int(cfg["run"].get("save_every_steps", 200))
    log_every = int(cfg["run"].get("log_every_steps", 20))
    clip = float(cfg["train"].get("grad_clip", 1.0))
    pad_id = tokenizer.pad_token_id
    max_len = int(cfg["data"]["max_len"])
    cap_steps = args.max_steps
    started = time.time()
    model.train()

    def do_save(pos, metrics):
        from peft import get_peft_model_state_dict
        return ckpt.save(
            run_dir / "checkpoints", position=pos,
            adapter_state=get_peft_model_state_dict(model),
            optimizer_state=optimizer.state_dict(), scheduler_state=None,
            rng=determinism.RngState.capture(), plan_hash=plan_hash,
            plan_summary={"epochs": [p.summary() for p in plans],
                          "total_steps": total_steps},
            mix_sha256=data.mix_sha256, config=cfg, metrics=metrics,
            keep_last=int(cfg["run"].get("keep_last_checkpoints", 3)))

    try:
        for epoch in range(position.epoch, epochs):
            steps = plans[epoch].steps
            start_at = position.step_in_epoch if epoch == position.epoch else 0
            for si in range(start_at, len(steps)):
                if cap_steps is not None and position.global_step >= cap_steps:
                    print(f"reached --max-steps {cap_steps}")
                    do_save(position, {"stopped": "max_steps"})
                    return
                step = steps[si]
                lr = lr_at(position.global_step, base_lr=float(cfg["train"]["lr"]),
                           total=total_steps, warmup=int(cfg["train"].get("warmup_steps", 0)),
                           schedule=cfg["train"].get("schedule", "cosine"),
                           min_ratio=float(cfg["train"].get("min_lr_ratio", 0.0)))
                for group in optimizer.param_groups:
                    group["lr"] = lr

                optimizer.zero_grad(set_to_none=True)
                loss_total, counted_total = 0.0, 0
                for micro in step.micro:
                    ii, am, lb = ds.collate(data, micro.indices, pad_id, max_len)
                    ii = torch.from_numpy(ii).to(device, non_blocking=True)
                    am = torch.from_numpy(am).to(device, non_blocking=True)
                    lb = torch.from_numpy(lb).to(device, non_blocking=True)
                    loss, loss_sum, counted = forward_loss(
                        model, ii, am, lb,
                        normaliser=float(step.supervised_tokens),
                        autocast_dtype=autocast_dtype, device_type=device.type)
                    loss.backward()
                    loss_total += float(loss_sum)
                    counted_total += counted
                    del ii, am, lb, loss, loss_sum

                if counted_total != step.supervised_tokens:
                    raise SystemExit(
                        f"supervised-token mismatch at step {position.global_step}: "
                        f"plan says {step.supervised_tokens}, data gave "
                        f"{counted_total}. The plan and the tokenised data "
                        f"disagree; refusing to continue.")

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], clip)
                optimizer.step()

                position = ckpt.Position(
                    epoch=epoch, step_in_epoch=si + 1,
                    global_step=position.global_step + 1,
                    rows_seen=position.rows_seen + step.rows,
                    supervised_tokens_seen=position.supervised_tokens_seen
                    + step.supervised_tokens)
                metrics = {
                    "loss": loss_total / max(counted_total, 1),
                    "lr": lr, "grad_norm": float(grad_norm),
                    "rows": step.rows, "supervised_tokens": step.supervised_tokens,
                    "elapsed_s": round(time.time() - started, 2),
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": position.global_step,
                                             "epoch": epoch, **metrics,
                                             **mem.snapshot()}) + "\n")
                if position.global_step % log_every == 0 or si == 0:
                    print(f"  step {position.global_step:>6,d}/{total_steps:,d} "
                          f"loss {metrics['loss']:.4f} lr {lr:.2e} "
                          f"gnorm {metrics['grad_norm']:.2f} "
                          f"rows {position.rows_seen:,d}")

                if _STOP["requested"]:
                    p = do_save(position, {**metrics, "stopped": "signal"})
                    print(f"checkpointed to {p.name}; resume with --resume auto")
                    return
                if position.global_step % save_every == 0:
                    do_save(position, metrics)

            position = ckpt.Position(epoch=epoch + 1, step_in_epoch=0,
                                     global_step=position.global_step,
                                     rows_seen=position.rows_seen,
                                     supervised_tokens_seen=position.supervised_tokens_seen)

        final = do_save(position, {"stopped": "complete"})
        print(f"\ntraining complete: {position.global_step:,d} steps, "
              f"{position.rows_seen:,d} rows, "
              f"{position.supervised_tokens_seen:,d} supervised tokens")
        print(f"final checkpoint: {final}")

    except torch.cuda.OutOfMemoryError as exc:
        path = do_save(position, {"stopped": "cuda_oom"})
        print("\n" + "=" * 72)
        print("CUDA OOM despite a passing pre-flight probe.")
        print(f"  state saved at {path.name} (step {position.global_step:,d})")
        print(f"  {mem.snapshot()}")
        print("The probe validated the widest micro-batch, so this is almost "
              "certainly external:\n"
              "  - another process took VRAM mid-run (check nvidia-smi), or\n"
              "  - allocator fragmentation.\n"
              "Free the card and `--resume auto` to replay this step bit-exactly.\n"
              "If it recurs, lower memory.max_tokens_per_microbatch -- but note "
              "that changes the plan, so it starts a NEW run, not a resume.")
        print("=" * 72)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
