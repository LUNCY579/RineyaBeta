#!/usr/bin/env python3
"""Train one real-model PDRR arm and export a scar-free compact checkpoint.

This harness is intentionally expensive: an optimizer micro-batch may execute an
intact teacher, an ablated teacher, a scheduled student, and an exactly compact
shadow student. The deployed checkpoint contains only the physically pruned
student; all gates, probes, frozen anchors, and Recovery Matrix parameters are
removed before export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(ROOT))

from pdrr.arms import get_arm  # noqa: E402
from pdrr.data import collate, deterministic_order, load_chat_jsonl, sha256_file  # noqa: E402
from pdrr.hf import (  # noqa: E402
    TraceCollector,
    freeze_except_recipients,
    install_layer_gate,
    install_recipient_probes,
    load_partial_state_dict,
    recipient_deltas,
    recipient_layers,
    remove_recipient_probes,
    resolve_decoder_layers,
    scar_free_check,
    set_layer_gate,
    set_probe_capture,
    trainable_state_dict,
)
from pdrr.losses import (  # noqa: E402
    anchor_preservation_loss,
    causal_significance,
    component_recovery_loss,
    feature_distillation_loss,
    token_kl,
)
from pdrr.provenance import (  # noqa: E402
    atomic_write_json,
    config_hash,
    environment_fingerprint,
    seed_everything,
)
from pdrr.routing import build_router, router_regularization  # noqa: E402
from pdrr.schedules import RecoverySchedule  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--target-layer", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    return payload


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def choose_device(request: str) -> torch.device:
    if request == "cpu":
        return torch.device("cpu")
    if request.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"requested {request}, but CUDA is unavailable")
        return torch.device(request)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dtype_for(name: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name!r}") from exc


def make_plan(size: int, total_steps: int, micro_batch: int, accumulation: int,
              seed: int) -> list[list[int]]:
    needed = total_steps * accumulation
    plan: list[list[int]] = []
    epoch = 0
    while len(plan) < needed:
        order = deterministic_order(size, seed=seed, epoch=epoch)
        for start in range(0, size, micro_batch):
            selected = order[start:start + micro_batch]
            if selected:
                plan.append(selected)
                if len(plan) == needed:
                    break
        epoch += 1
    return plan


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temp_name)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def latest_checkpoint(directory: Path) -> Path | None:
    candidates = sorted(directory.glob("step_*.pt"))
    return candidates[-1] if candidates else None


def checkpoint_payload(
    *,
    step: int,
    student: torch.nn.Module,
    router: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    config_digest: str,
    plan_digest: str,
    data_digest: str,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "step": step,
        "student_trainable_state": trainable_state_dict(student),
        "router_state": None if router is None else router.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        "config_hash": config_digest,
        "plan_hash": plan_digest,
        "data_sha256": data_digest,
        "environment": environment_fingerprint(),
    }


def restore_checkpoint(
    path: Path,
    *,
    student: torch.nn.Module,
    router: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    config_digest: str,
    plan_digest: str,
    data_digest: str,
) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for name, expected in (("config_hash", config_digest), ("plan_hash", plan_digest),
                           ("data_sha256", data_digest)):
        if payload.get(name) != expected:
            raise RuntimeError(f"checkpoint {name} mismatch; refusing non-reproducible resume")
    load_partial_state_dict(student, payload["student_trainable_state"])
    if router is not None and payload.get("router_state") is not None:
        router.load_state_dict(payload["router_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    random.setstate(payload["rng"]["python"])
    np.random.set_state(payload["rng"]["numpy"])
    torch.set_rng_state(payload["rng"]["torch"])
    if payload["rng"].get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
    return int(payload["step"])


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    arm = get_arm(args.arm)
    seed = int(args.seed if args.seed is not None else config.get("seed", 20260823))
    seed_everything(seed)

    output = resolve_path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    student_device = choose_device(str(config.get("device", "auto")))
    teacher_request = str(config["model"].get("teacher_device", "cpu"))
    teacher_device = student_device if teacher_request == "same" else choose_device(teacher_request)
    model_path = resolve_path(str(config["model"]["path"]))
    data_path = resolve_path(str(config["data"]["train"]))
    if not model_path.exists():
        raise FileNotFoundError(f"model path not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"training data not found: {data_path}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    student_dtype = dtype_for(str(config["model"].get("dtype", "bfloat16")), student_device)
    teacher_dtype = dtype_for(str(config["model"].get("dtype", "bfloat16")), teacher_device)
    attention_impl = str(config["model"].get("attn_implementation", "eager"))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    student = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=student_dtype, attn_implementation=attention_impl)
    teacher = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=teacher_dtype, attn_implementation=attention_impl)
    student.config.use_cache = False
    teacher.config.use_cache = False
    student.to(student_device)
    teacher.to(teacher_device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    _, layers = resolve_decoder_layers(student)
    target = int(args.target_layer)
    recipients = recipient_layers(
        target, len(layers), radius=int(config["recipients"].get("radius", 2)),
        max_count=int(config["recipients"].get("max_count", 3)))
    if not recipients:
        raise RuntimeError("recipient selection is empty")

    install_layer_gate(student, target, 1.0 if arm.deletion != "hard" else 0.0)
    install_layer_gate(teacher, target, 1.0)
    trainable = freeze_except_recipients(student, recipients)
    probes = install_recipient_probes(student, recipients)

    router = build_router(
        arm.routing,
        state_dim=int(student.config.hidden_size),
        recipient_layers=recipients,
        source_layer=target,
        seed=seed + 991,
        hidden_dim=int(config["routing"].get("hidden_dim", 128)),
        temperature=float(config["routing"].get("temperature", 0.8)),
        top_k=int(config["routing"].get("top_k", len(recipients))),
        max_positions=int(config["data"].get("max_length", 384)),
    )
    if router is not None:
        router.to(student_device)
        if arm.router_trainable:
            trainable.extend([parameter for parameter in router.parameters()
                              if parameter.requires_grad])
        else:
            for parameter in router.parameters():
                parameter.requires_grad_(False)

    examples = load_chat_jsonl(
        data_path, tokenizer, max_length=int(config["data"].get("max_length", 384)),
        max_rows=config["data"].get("max_rows"))
    requested_steps = int(config["training"].get("steps", 300))
    total_steps = min(requested_steps, args.max_steps) if args.max_steps else requested_steps
    micro_batch = int(config["training"].get("micro_batch_size", 1))
    accumulation = int(config["training"].get("gradient_accumulation", 1))
    plan = make_plan(len(examples), total_steps, micro_batch, accumulation, seed)
    plan_digest = hashlib.sha256(json.dumps(plan, separators=(",", ":")).encode()).hexdigest()
    config_digest = config_hash({"config": config, "arm": arm.name, "target": target,
                                 "seed": seed, "recipients": recipients,
                                 "total_steps": total_steps})
    data_digest = sha256_file(data_path)

    run_manifest = {
        "schema_version": 1,
        "arm": arm.name,
        "arm_definition": arm.__dict__,
        "target_layer": target,
        "recipients": recipients,
        "seed": seed,
        "model_path": str(model_path),
        "data_path": str(data_path),
        "data_sha256": data_digest,
        "config_path": str(config_path),
        "config_hash": config_digest,
        "plan_hash": plan_digest,
        "student_device": str(student_device),
        "teacher_device": str(teacher_device),
        "total_steps": total_steps,
        "temporary_router_parameters": 0 if router is None else sum(p.numel() for p in router.parameters()),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "environment": environment_fingerprint(),
    }
    atomic_write_json(output / "run_manifest.json", run_manifest)

    if args.dry_run:
        print(json.dumps(run_manifest, indent=2, default=str))
        return

    if not arm.train:
        set_probe_capture(probes, False)
        remove_recipient_probes(student, probes)
        if arm.name == "full":
            export = output / "export_full"
            student.save_pretrained(export, safe_serialization=True)
            tokenizer.save_pretrained(export)
            atomic_write_json(output / "export_manifest.json", {
                "arm": arm.name, "compact": False,
                "parameter_count": sum(p.numel() for p in student.parameters()),
            })
            print(f"exported intact anchor to {export}")
            return
        set_layer_gate(student, target, 0.0)
        sample = collate(examples[:min(2, len(examples))], tokenizer.pad_token_id)
        sample = {key: value.to(student_device) for key, value in sample.items()
                  if torch.is_tensor(value) and key != "labels"}
        compact, report = scar_free_check(student, sample, target)
        if not report["passed"]:
            raise RuntimeError(f"scar-free export check failed: {report}")
        export = output / "export_compact"
        compact.save_pretrained(export, safe_serialization=True)
        tokenizer.save_pretrained(export)
        atomic_write_json(output / "export_manifest.json", {"arm": arm.name, **report})
        print(f"exported hard-pruned anchor to {export}")
        return

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["training"].get("learning_rate", 1e-4)),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    warmup = round(total_steps * float(config["training"].get("warmup_fraction", 0.05)))
    schedule = RecoverySchedule(
        total_steps,
        notice_fraction=float(config["training"].get("notice_fraction", 0.20)),
        withdrawal_end_fraction=float(config["training"].get("withdrawal_end_fraction", 0.70)),
        recovery_end_fraction=float(config["training"].get("recovery_end_fraction", 0.78)),
    )
    start_step = 0
    if args.resume:
        checkpoint = latest_checkpoint(checkpoint_dir)
        if checkpoint is not None:
            start_step = restore_checkpoint(
                checkpoint, student=student, router=router, optimizer=optimizer,
                config_digest=config_digest, plan_digest=plan_digest,
                data_digest=data_digest)
            print(f"resumed from {checkpoint.name} at optimizer step {start_step}")

    temperature = float(config["training"].get("teacher_temperature", 2.0))
    lambda_lm = float(config["training"].get("lambda_lm", 0.35))
    lambda_kd = float(config["training"].get("lambda_kd", 0.65))
    lambda_feature = float(config["training"].get("lambda_feature", 0.20))
    lambda_recovery = float(config["training"].get("lambda_recovery", 0.80))
    lambda_anchor = float(config["training"].get("lambda_anchor", 0.10))
    clip = float(config["training"].get("gradient_clip", 1.0))
    save_every = int(config.get("checkpoint", {}).get("save_every_steps", 25))
    log_path = output / "train_log.jsonl"
    started = time.time()
    last_sample: dict[str, torch.Tensor] | None = None

    for step in range(start_step, total_steps):
        if step < warmup and warmup > 0:
            lr_scale = (step + 1) / warmup
        else:
            progress = (step - warmup) / max(total_steps - warmup, 1)
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        for group in optimizer.param_groups:
            group["lr"] = float(config["training"].get("learning_rate", 1e-4)) * lr_scale

        optimizer.zero_grad(set_to_none=True)
        step_metrics: dict[str, float] = {}
        for micro in range(accumulation):
            selected = [examples[index] for index in plan[step * accumulation + micro]]
            batch_cpu = collate(selected, tokenizer.pad_token_id)
            student_batch = {key: value.to(student_device) for key, value in batch_cpu.items()
                             if torch.is_tensor(value)}
            teacher_batch = {key: value.to(teacher_device) for key, value in batch_cpu.items()
                             if torch.is_tensor(value) and key != "labels"}
            mask = student_batch["attention_mask"]

            set_layer_gate(teacher, target, 1.0)
            with torch.no_grad(), TraceCollector(teacher, [target], detach=True) as collector:
                teacher_full = teacher(**teacher_batch, output_hidden_states=True)
            teacher_trace = collector.trace()
            set_layer_gate(teacher, target, 0.0)
            with torch.no_grad():
                teacher_ablated = teacher(**teacher_batch, output_hidden_states=True)
            full_logits = teacher_full.logits.to(student_device)
            ablated_logits = teacher_ablated.logits.to(student_device)
            teacher_hidden = teacher_full.hidden_states[-1].to(student_device)
            source_residual = teacher_trace.residuals[target].to(student_device)
            significance = causal_significance(full_logits, ablated_logits)
            significance = significance * mask.to(significance.dtype)

            gate = 0.0 if arm.deletion == "hard" else schedule.doomed_gate(step)
            set_layer_gate(student, target, gate)
            set_probe_capture(probes, False)
            with TraceCollector(student, [target], detach=False):
                scheduled = student(**student_batch, output_hidden_states=True)

            set_layer_gate(student, target, 0.0)
            set_probe_capture(probes, True)
            with TraceCollector(student, [target], detach=False) as compact_collector:
                compact = student(**student_batch, output_hidden_states=True)
            compact_trace = compact_collector.trace()
            deltas = recipient_deltas(probes)

            zero = scheduled.logits.new_zeros(())
            supervised = scheduled.loss if arm.supervised else zero
            kd = masked_mean(token_kl(full_logits, scheduled.logits, temperature), mask) if arm.kd else zero
            compact_kd = masked_mean(token_kl(full_logits, compact.logits, temperature), mask) if arm.kd else zero
            feature = feature_distillation_loss(teacher_hidden, scheduled.hidden_states[-1]) if arm.feature_kd else zero
            recovery = zero
            anchor = zero
            router_reg = zero
            route_diagnostics: dict[str, float] = {}

            if router is not None:
                source_state = compact_trace.inputs[target]
                positions = torch.arange(source_state.shape[1], device=student_device)
                route = router(source_state, positions)
                reg = router_regularization(
                    route,
                    entropy_weight=float(config["routing"].get("entropy_weight", 0.002)),
                    load_balance_weight=float(config["routing"].get("load_balance_weight", 0.01)),
                )
                router_reg = reg.total
                mean_route = route.float().mean(dim=(0, 1)).detach().cpu().tolist()
                route_diagnostics = {
                    "route_entropy": float((-(route.float() * route.float().clamp_min(1e-12).log())
                                             .sum(dim=-1).mean()).detach()),
                    "route_max": float(route.max().detach()),
                    **{f"route_mean_layer_{layer}": float(mass)
                       for layer, mass in zip(recipients, mean_route)},
                }
                if arm.recovery_loss and schedule.recovery_weight(step) > 0:
                    recovery, _ = component_recovery_loss(
                        source_residual,
                        deltas,
                        route,
                        significance if arm.causal_significance else None,
                    )
                    current_map = {layer: delta for layer, delta in zip(sorted(probes), deltas)}
                    zero_map = {layer: torch.zeros_like(delta) for layer, delta in current_map.items()}
                    anchor = anchor_preservation_loss(
                        current_map, zero_map, recipient_layers=sorted(probes))

            recovery_scale = schedule.recovery_weight(step)
            compact_scale = 1.0 if schedule.compact_only(step) else 0.25 + 0.75 * (1.0 - gate)
            total = (
                lambda_lm * supervised
                + lambda_kd * kd
                + lambda_kd * compact_scale * compact_kd
                + lambda_feature * feature
                + lambda_recovery * recovery_scale * recovery
                + lambda_anchor * recovery_scale * anchor
                + router_reg
            ) / accumulation
            if not torch.isfinite(total):
                raise RuntimeError(f"non-finite loss at step {step}, microbatch {micro}")
            total.backward()
            values = {
                "total": float(total.detach()) * accumulation,
                "supervised": float(supervised.detach()),
                "kd": float(kd.detach()),
                "compact_kd": float(compact_kd.detach()),
                "feature": float(feature.detach()),
                "recovery": float(recovery.detach()),
                "anchor": float(anchor.detach()),
                "router_regularization": float(router_reg.detach()),
                **route_diagnostics,
            }
            for key, value in values.items():
                step_metrics[key] = step_metrics.get(key, 0.0) + value / accumulation
            last_sample = {key: value[:min(2, value.shape[0])].detach()
                           for key, value in student_batch.items()
                           if key in {"input_ids", "attention_mask"}}

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, clip)
        optimizer.step()
        record = {
            "step": step + 1,
            "phase": schedule.phase(step),
            "doomed_gate": 0.0 if arm.deletion == "hard" else schedule.doomed_gate(step),
            "recovery_scale": schedule.recovery_weight(step),
            "lr": optimizer.param_groups[0]["lr"],
            "grad_norm": float(grad_norm),
            "elapsed_s": round(time.time() - started, 2),
            **step_metrics,
        }
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if (step + 1) % 10 == 0 or step == start_step:
            print(f"step {step + 1:>5}/{total_steps} phase={record['phase']:<13} "
                  f"gate={record['doomed_gate']:.3f} loss={record['total']:.4f}")
        if (step + 1) % save_every == 0 or step + 1 == total_steps:
            payload = checkpoint_payload(
                step=step + 1, student=student, router=router, optimizer=optimizer,
                config_digest=config_digest, plan_digest=plan_digest,
                data_digest=data_digest)
            atomic_torch_save(checkpoint_dir / f"step_{step + 1:09d}.pt", payload)

    if last_sample is None:
        raise RuntimeError("training produced no sample for export verification")
    set_layer_gate(student, target, 0.0)
    set_probe_capture(probes, False)
    remove_recipient_probes(student, probes)
    compact, report = scar_free_check(student, last_sample, target)
    if not report["passed"]:
        raise RuntimeError(f"scar-free export check failed: {report}")
    export = output / "export_compact"
    compact.save_pretrained(export, safe_serialization=True)
    tokenizer.save_pretrained(export)
    atomic_write_json(output / "export_manifest.json", {
        "arm": arm.name,
        "target_layer": target,
        "recipients": recipients,
        "router_deleted": True,
        "temporary_probes_deleted": True,
        **report,
    })
    print(f"exported scar-free compact checkpoint to {export}")


if __name__ == "__main__":
    main()
