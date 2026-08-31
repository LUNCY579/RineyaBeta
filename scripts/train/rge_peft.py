#!/usr/bin/env python3
"""RGE-PEFT experimental trainer: M1 AdamW parity plus delta observation.

This is intentionally a narrow harness around the established ``heal.py``
substrate.  Model construction, loss, data, batching, determinism, memory probing,
and checkpoint primitives are reused verbatim.  M1 adds only an explicit optimizer
factory, resolved provenance, and detached effective-LoRA-delta measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

TRAIN_DIR = Path(__file__).resolve().parent
ROOT = TRAIN_DIR.parents[1]
sys.path.insert(0, str(TRAIN_DIR))

import checkpoint as ckpt  # noqa: E402
import dataset as ds  # noqa: E402
import determinism  # noqa: E402
import heal  # noqa: E402
import memory as mem  # noqa: E402
from optim.delta_geometry import (  # noqa: E402
    DeltaGeometryTracker,
    append_geometry,
    reconcile_geometry_log,
)
from optim.factory import build_optimizer, optimizer_contract  # noqa: E402
from plan import build_plan  # noqa: E402

REQUIRED_SECTIONS = {
    "program_version", "seed", "model", "data", "lora", "optimizer",
    "geometry", "schedule", "curriculum", "cma", "train", "memory",
    "determinism", "resume", "run",
}
_STOP = {"requested": False, "signal": None}


def _request_stop(signum, _frame) -> None:
    if _STOP["requested"]:
        raise SystemExit(130)
    _STOP["requested"] = True
    _STOP["signal"] = signum
    print(f"\nsignal {signum} received: finish step, checkpoint, then exit")


def load_rge_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_SECTIONS - set(cfg or {}))
    if missing:
        raise ValueError(f"RGE config is missing required sections: {missing}")
    for key in ("allow_env_drift", "allow_code_drift"):
        if key not in cfg["resume"]:
            raise ValueError(f"resume.{key} must be explicit")
    if cfg["resume"]["allow_env_drift"] or cfg["resume"]["allow_code_drift"]:
        raise ValueError("RGE configs must default to strict environment/code drift refusal")
    if cfg["curriculum"].get("mode") != "reference":
        raise ValueError("M1 permits only curriculum.mode=reference")
    if bool(cfg["cma"].get("enabled")):
        raise ValueError("CMA is gated by M5 and cannot be enabled in M1")
    cfg["_config_path"] = str(path.resolve())
    return cfg


def as_legacy_config(cfg: dict) -> dict:
    """Translate the explicit RGE namespace to the reference trainer contract."""
    opt, schedule = cfg["optimizer"], cfg["schedule"]
    train = cfg["train"]
    return {
        "seed": int(cfg["seed"]),
        "device": cfg.get("device", "auto"),
        "run": dict(cfg["run"]),
        "model": dict(cfg["model"]),
        "data": dict(cfg["data"]),
        "lora": dict(cfg["lora"]),
        "train": {
            "epochs": int(train["epochs"]),
            "lr": float(opt["lr"]),
            "schedule": str(schedule["name"]),
            "min_lr_ratio": float(schedule["min_lr_ratio"]),
            "warmup_steps": int(schedule["warmup_steps"]),
            "weight_decay": float(opt["weight_decay"]),
            "grad_clip": float(train["grad_clip"]),
            "grad_accum": int(train["grad_accum"]),
            "gradient_checkpointing": bool(train["gradient_checkpointing"]),
        },
        "memory": dict(cfg["memory"]),
        "determinism": dict(cfg["determinism"]),
        "resume": dict(cfg["resume"]),
        "_config_path": cfg["_config_path"],
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_files(root: Path) -> dict:
    files = {}
    for path in sorted((p for p in root.rglob("*") if p.is_file()),
                       key=lambda p: p.relative_to(root).as_posix()):
        files[path.relative_to(root).as_posix()] = _sha256_file(path)
    combined = hashlib.sha256("".join(
        f"{name}:{value}" for name, value in sorted(files.items())).encode()).hexdigest()
    return {"combined_sha256": combined, "files": files}


def _tensor_hashes(model) -> dict:
    files = {}
    for name, param in sorted(model.named_parameters(), key=lambda item: item[0]):
        if not param.requires_grad:
            continue
        value = param.detach().to(device="cpu").contiguous()
        digest = hashlib.sha256(value.numpy().tobytes()).hexdigest()
        files[name] = {"sha256": digest, "shape": list(value.shape),
                       "dtype": str(value.dtype), "numel": value.numel()}
    combined = hashlib.sha256("".join(
        f"{name}:{payload['sha256']}" for name, payload in sorted(files.items())
    ).encode()).hexdigest()
    return {"combined_sha256": combined, "tensors": files,
            "parameter_count": int(sum(v["numel"] for v in files.values()))}


def _git(args: list[str]) -> str:
    proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
                   encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _first_batches_hash(plans, count: int = 3) -> str:
    payload = []
    for plan in plans:
        for step in plan.steps:
            payload.append([[*micro.indices] for micro in step.micro])
            if len(payload) >= count:
                return hashlib.sha256(json.dumps(payload, separators=(",", ":"))
                                      .encode()).hexdigest()
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _manifest(cfg: dict, legacy: dict, model, tokenizer, data, plans,
              plan_hash: str, contract: dict, geometry: DeltaGeometryTracker,
              device: torch.device) -> dict:
    model_dir = ROOT / legacy["model"]["path"]
    eval_manifest = json.loads((ROOT / "manifests/eval_assets.json").read_text(
        encoding="utf-8"))
    status = _git(["status", "--porcelain"])
    tracked_status = _git(["status", "--porcelain", "--untracked-files=no"])
    try:
        import peft
        import transformers
    except ImportError:  # pragma: no cover - model construction already requires these
        peft = transformers = None
    cuda = torch.cuda.is_available()
    return {
        "schema_version": 1,
        "run_id": str(cfg["run"]["id"]),
        "program_version": str(cfg["program_version"]),
        "designation": str(cfg["run"]["designation"]),
        "git": {"branch": _git(["branch", "--show-current"]),
                "commit": _git(["rev-parse", "HEAD"]),
                "origin_main": _git(["rev-parse", "origin/main"]),
                "origin_research": _git(["rev-parse", "origin/research/rge-peft-v1"]),
                "dirty": bool(status), "tracked_dirty": bool(tracked_status),
                "status_porcelain": status.splitlines()},
        "model": {"path": _relative(model_dir), **_combined_files(model_dir)},
        "tokenizer": {
            "path": _relative(model_dir / "tokenizer.json"),
            "sha256": _sha256_file(model_dir / "tokenizer.json"),
            "class": tokenizer.__class__.__name__,
        },
        "data": {"mix": _relative(ROOT / legacy["data"]["mix"]),
                 "mix_sha256": data.mix_sha256, "cache_key": data.cache_key,
                 "rows": len(data)},
        "evaluation_assets": {
            "manifest_sha256": _sha256_file(ROOT / "manifests/eval_assets.json"),
            "sets": {name: {"path": value["path"], "rows": value["rows"],
                             "sha256": value["sha256"], "note": value.get("note")}
                     for name, value in eval_manifest["eval_sets"].items()},
            "sealed_accessed": False,
        },
        "seed": int(cfg["seed"]),
        "environment": {**determinism.fingerprint(),
                        "python": sys.version,
                        "run_device": str(device),
                        "requested_device": cfg.get("device", "auto"),
                        "transformers": getattr(transformers, "__version__", None),
                        "peft": getattr(peft, "__version__", None)},
        "precision": legacy["model"]["dtype"],
        "lora": dict(cfg["lora"]),
        "trainable_initial_state": _tensor_hashes(model),
        "optimizer": {**dict(cfg["optimizer"]), **contract},
        "geometry": geometry.contract,
        "schedule": dict(cfg["schedule"]),
        "curriculum": dict(cfg["curriculum"]),
        "cma": dict(cfg["cma"]),
        "supervised_token_budget": cfg["train"]["supervised_token_budget"],
        "plan_hash": plan_hash,
        "plan_summaries": [plan.summary() for plan in plans],
        "first_batches_sha256": _first_batches_hash(plans),
        "memory": dict(cfg["memory"]),
        "source_fingerprint": determinism.code_fingerprint(),
        "wall_time_s": None, "active_gpu_hours": 0.0,
        "peak_allocated_mib": None, "peak_reserved_mib": None,
        "exit_reason": "running", "resume_events": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="auto", help="auto | none | path")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--geometry", choices=("config", "on", "off"), default="config")
    args = parser.parse_args()

    cfg = load_rge_config(Path(args.config))
    if args.geometry != "config":
        cfg["geometry"]["enabled"] = args.geometry == "on"
    legacy = as_legacy_config(cfg)
    seed = int(cfg["seed"])
    determinism.enable(seed, strict=bool(cfg["determinism"]["strict"]))

    use_cuda = torch.cuda.is_available() and cfg.get("device", "auto") != "cpu"
    device = torch.device("cuda:0" if use_cuda else "cpu")
    run_dir = Path(args.out) if args.out else ROOT / cfg["run"]["dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = heal.take_run_lock(run_dir)
    manifest_path = run_dir / "run_manifest.json"
    overall_started = time.perf_counter()
    training_seconds = 0.0
    manifest = None

    try:
        if args.resume == "none" and (run_dir / "checkpoints/latest.json").exists():
            raise SystemExit("--resume none refuses a run directory that already contains "
                             "checkpoints; choose a new --out directory")
        cap = mem.apply_cap(cfg["memory"]["cap_mib"] if use_cuda else None)
        print(f"device={device}" + (f" cap={cap:,.0f} MiB" if cap else ""))
        tokenizer, model = heal.build_model(legacy, device)
        mix_path = ROOT / legacy["data"]["mix"]
        data = ds.build(mix_path, tokenizer, max_len=int(legacy["data"]["max_len"]),
                        cache_dir=run_dir / "cache",
                        tokenizer_dir=ROOT / legacy["model"]["path"])
        plans = [build_plan(
            data.total_lengths, data.supervised_lengths, seed=seed, epoch=epoch,
            max_tokens_per_microbatch=int(legacy["memory"]["max_tokens_per_microbatch"]),
            max_rows_per_microbatch=int(legacy["memory"]["max_rows_per_microbatch"]),
            micro_batches_per_step=int(legacy["train"]["grad_accum"]),
            megabatch_rows=int(legacy["data"]["megabatch_rows"]),
            max_len=int(legacy["data"]["max_len"]),
            width_bucket=int(legacy["data"].get("width_bucket", 1)))
            for epoch in range(int(legacy["train"]["epochs"]))]
        total_steps = sum(len(plan.steps) for plan in plans)
        plan_hash = ";".join(plan.hash() for plan in plans)
        print(f"plan: {len(plans)} epoch(s), {total_steps:,d} optimizer steps")

        optimizer = build_optimizer(model, cfg["optimizer"])
        opt_contract = optimizer_contract(model, cfg["optimizer"])
        geometry = DeltaGeometryTracker(
            model, enabled=bool(cfg["geometry"]["enabled"]),
            cadence_steps=int(cfg["geometry"]["cadence_steps"]),
            include_radius_angle=bool(cfg["geometry"]["include_radius_angle"]),
            run_id=str(cfg["run"]["id"]))
        manifest = _manifest(cfg, legacy, model, tokenizer, data, plans, plan_hash,
                             opt_contract, geometry, device)
        _atomic_json(manifest_path, manifest)
        (run_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")},
                           sort_keys=False), encoding="utf-8", newline="\n")

        by_shape = {}
        for plan in plans:
            for step in plan.steps:
                for micro in step.micro:
                    by_shape[(len(micro.indices), micro.max_len)] = micro.padded_cost
        accum = int(legacy["train"]["grad_accum"])
        probe_shapes = sorted(by_shape, key=lambda key: -by_shape[key])[:accum]
        while len(probe_shapes) < accum:
            probe_shapes.append(probe_shapes[0])
        autocast_dtype = (torch.bfloat16 if use_cuda
                          and legacy["model"]["dtype"] == "bfloat16" else None)
        model.train()

        def probe_step(batches) -> None:
            normaliser = float(sum(rows * width for rows, width in probe_shapes))
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                for input_ids, attention, labels in batches:
                    loss, _, _ = heal.forward_loss(
                        model, input_ids, attention, labels, normaliser=normaliser,
                        autocast_dtype=autocast_dtype, device_type=device.type)
                    loss.backward()
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        before = {name: param.detach().clone() for name, param in model.named_parameters()
                  if param.requires_grad}
        report = mem.probe(probe_step, shapes=probe_shapes, cap_mib=cap,
                           margin_mib=float(cfg["memory"]["margin_mib"]),
                           enabled=use_cuda)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.copy_(before[name])
        del before
        optimizer.state.clear()
        optimizer.zero_grad(set_to_none=True)
        determinism.seed_everything(seed)
        print(report.render())
        _atomic_json(run_dir / "memory_profile.json", report.__dict__)
        if not report.ok:
            raise SystemExit("memory probe failed; refusing to train")

        position = ckpt.Position(epoch=0, step_in_epoch=0, global_step=0,
                                 rows_seen=0, supervised_tokens_seen=0)
        resumed_from = None
        if args.resume != "none":
            path = (ckpt.latest_path(run_dir / "checkpoints") if args.resume == "auto"
                    else Path(args.resume))
            if path and path.exists():
                payload = ckpt.load(
                    path, expect_plan_hash=plan_hash, expect_mix_sha256=data.mix_sha256,
                    allow_env_drift=False, allow_code_drift=False)
                if payload.get("format_version") != 2 or "extra_state" not in payload:
                    raise SystemExit("checkpoint is not an RGE format-2 checkpoint")
                rge_state = payload["extra_state"].get("rge", {})
                if rge_state.get("optimizer_contract") != opt_contract:
                    raise SystemExit("optimizer routing/implementation changed across resume")
                from peft import set_peft_model_state_dict
                set_peft_model_state_dict(model, payload["adapter_state"])
                optimizer.load_state_dict(payload["optimizer_state"])
                geometry.load_state_dict(rge_state["geometry_state"])
                payload["rng_state"].restore()
                position = payload["position_obj"]
                resumed_from = _relative(path)
                manifest["resume_events"].append({"checkpoint": resumed_from,
                                                   "global_step": position.global_step})
                print(f"resumed {path.name} at step {position.global_step}")

        geometry_path = run_dir / str(cfg["geometry"]["artifact"])
        if resumed_from:
            reconcile_geometry_log(geometry_path, through_step=position.global_step)
        elif geometry_path.exists():
            geometry_path.unlink()
        _atomic_json(manifest_path, manifest)

        if args.dry_run:
            manifest["exit_reason"] = "dry_run"
            return

        for signame in ("SIGINT", "SIGTERM", "SIGBREAK"):
            signum = getattr(signal, signame, None)
            if signum is not None:
                try:
                    signal.signal(signum, _request_stop)
                except (OSError, ValueError):
                    pass

        log_path = run_dir / "train_log.jsonl"
        save_every = int(cfg["run"]["save_every_steps"])
        log_every = int(cfg["run"]["log_every_steps"])
        clip = float(cfg["train"]["grad_clip"])
        pad_id = tokenizer.pad_token_id
        max_len = int(cfg["data"]["max_len"])
        budget = cfg["train"]["supervised_token_budget"]
        model.train()
        last_metrics = {}

        def do_save(pos, metrics):
            from peft import get_peft_model_state_dict
            saved = ckpt.save(
                run_dir / "checkpoints", position=pos,
                adapter_state=get_peft_model_state_dict(model),
                optimizer_state=optimizer.state_dict(), scheduler_state=None,
                rng=determinism.RngState.capture(), plan_hash=plan_hash,
                plan_summary={"epochs": [plan.summary() for plan in plans],
                              "total_steps": total_steps},
                mix_sha256=data.mix_sha256,
                config={k: v for k, v in cfg.items() if not k.startswith("_")},
                metrics=metrics,
                keep_last=int(cfg["run"]["keep_last_checkpoints"]),
                extra_state={"rge": {
                    "program_version": cfg["program_version"],
                    "optimizer_contract": opt_contract,
                    "geometry_contract": geometry.contract,
                    "geometry_state": geometry.state_dict(),
                    "curriculum": cfg["curriculum"], "cma": cfg["cma"],
                }})
            ledger_row = {
                "schema_version": 1, "checkpoint": saved.name,
                "global_step": pos.global_step, "position": pos.__dict__,
                "format_version": 2, "plan_hash": plan_hash,
                "mix_sha256": data.mix_sha256, "metrics": metrics,
            }
            with (run_dir / "checkpoint_ledger.jsonl").open(
                    "a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(ledger_row, sort_keys=True, allow_nan=False) + "\n")
            return saved

        exit_reason = "complete"
        for epoch in range(position.epoch, len(plans)):
            steps = plans[epoch].steps
            start_at = position.step_in_epoch if epoch == position.epoch else 0
            for step_index in range(start_at, len(steps)):
                if args.max_steps is not None and position.global_step >= args.max_steps:
                    exit_reason = "max_steps"
                    do_save(position, {**last_metrics, "stopped": exit_reason})
                    return
                if budget is not None and position.supervised_tokens_seen >= int(budget):
                    exit_reason = "supervised_token_budget"
                    do_save(position, {**last_metrics, "stopped": exit_reason})
                    return
                if use_cuda:
                    torch.cuda.synchronize()
                step_started = time.perf_counter()
                step = steps[step_index]
                lr = heal.lr_at(
                    position.global_step, base_lr=float(cfg["optimizer"]["lr"]),
                    total=total_steps, warmup=int(cfg["schedule"]["warmup_steps"]),
                    schedule=str(cfg["schedule"]["name"]),
                    min_ratio=float(cfg["schedule"]["min_lr_ratio"]))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                loss_total, counted_total = 0.0, 0
                for micro in step.micro:
                    ii, am, lb = ds.collate(data, micro.indices, pad_id, max_len)
                    ii = torch.from_numpy(ii).to(device, non_blocking=True)
                    am = torch.from_numpy(am).to(device, non_blocking=True)
                    lb = torch.from_numpy(lb).to(device, non_blocking=True)
                    loss, loss_sum, counted = heal.forward_loss(
                        model, ii, am, lb, normaliser=float(step.supervised_tokens),
                        autocast_dtype=autocast_dtype, device_type=device.type)
                    loss.backward()
                    loss_total += float(loss_sum)
                    counted_total += counted
                    del ii, am, lb, loss, loss_sum
                if counted_total != step.supervised_tokens:
                    raise SystemExit("supervised-token mismatch between plan and data")
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad], clip)
                optimizer.step()
                next_step = position.global_step + 1
                append_geometry(geometry_path, geometry.measure(next_step))
                if use_cuda:
                    torch.cuda.synchronize()
                step_seconds = time.perf_counter() - step_started
                training_seconds += step_seconds
                position = ckpt.Position(
                    epoch=epoch, step_in_epoch=step_index + 1, global_step=next_step,
                    rows_seen=position.rows_seen + step.rows,
                    supervised_tokens_seen=position.supervised_tokens_seen
                    + step.supervised_tokens)
                last_metrics = {
                    "loss": loss_total / max(counted_total, 1), "lr": lr,
                    "grad_norm": float(grad_norm), "rows": step.rows,
                    "supervised_tokens": step.supervised_tokens,
                    "step_time_s": step_seconds,
                }
                heal.beat(lock)
                with log_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps({"step": position.global_step,
                                             "epoch": epoch, **last_metrics,
                                             **mem.snapshot()}, allow_nan=False) + "\n")
                if position.global_step % log_every == 0 or step_index == 0:
                    print(f"step {position.global_step:>5}/{total_steps} "
                          f"loss {last_metrics['loss']:.4f} lr {lr:.2e} "
                          f"{step_seconds:.2f}s")
                if _STOP["requested"]:
                    exit_reason = "signal"
                    do_save(position, {**last_metrics, "stopped": exit_reason})
                    return
                if position.global_step % save_every == 0:
                    do_save(position, last_metrics)
            position = ckpt.Position(
                epoch=epoch + 1, step_in_epoch=0, global_step=position.global_step,
                rows_seen=position.rows_seen,
                supervised_tokens_seen=position.supervised_tokens_seen)
        do_save(position, {**last_metrics, "stopped": exit_reason})

    finally:
        lock.unlink(missing_ok=True)
        if manifest is not None:
            manifest["wall_time_s"] = round(time.perf_counter() - overall_started, 6)
            manifest["active_gpu_hours"] = (training_seconds / 3600.0
                                             if use_cuda else 0.0)
            snapshot = mem.snapshot() if use_cuda else {}
            probe_report = locals().get("report")
            probe_allocated = (probe_report.peak_alloc_mib if probe_report and
                               probe_report.ran else 0.0)
            probe_reserved = (probe_report.peak_reserved_mib if probe_report and
                              probe_report.ran else 0.0)
            manifest["peak_allocated_mib"] = max(
                float(snapshot.get("allocated_mib", 0.0)), probe_allocated)
            manifest["peak_reserved_mib"] = max(
                float(snapshot.get("peak_reserved_mib", 0.0)), probe_reserved)
            if manifest.get("exit_reason") == "running":
                # Infer the controlled return path from the last saved metrics.
                manifest["exit_reason"] = (locals().get("exit_reason") or
                                           ("dry_run" if args.dry_run else "error"))
            manifest["final_position"] = (position.__dict__ if "position" in locals()
                                          else None)
            _atomic_json(manifest_path, manifest)
            _atomic_json(run_dir / "summary.json", {
                "schema_version": 1, "run_id": manifest["run_id"],
                "exit_reason": manifest["exit_reason"],
                "final_position": manifest["final_position"],
                "wall_time_s": manifest["wall_time_s"],
                "active_gpu_hours": manifest["active_gpu_hours"],
                "peak_allocated_mib": manifest["peak_allocated_mib"],
                "peak_reserved_mib": manifest["peak_reserved_mib"],
                "last_metrics": locals().get("last_metrics", {}),
                "sealed_data_accessed": False,
            })


if __name__ == "__main__":
    main()
