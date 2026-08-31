#!/usr/bin/env python3
"""Pre-register and run the M2 AdamW-relative Muon effective-step bracket."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RGE = ROOT / "scripts" / "train" / "rge_peft.py"
PAIR = ROOT / "manifests" / "rge_peft" / "m2_optimizer_pair.yaml"
ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYTHONIOENCODING": "utf-8",
}
TARGET_RATIOS = (0.5, 1.0, 2.0)
TARGET_FACTOR_TOLERANCE = 1.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a quantile of no values")
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def lr_label(lr: float) -> str:
    return f"{lr:.1e}".replace("+", "").replace("-", "m").replace(".", "p")


def load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def calibration_config(source: dict, *, run_id: str, run_dir: Path,
                       lr: float | None) -> dict:
    config = copy.deepcopy(source)
    if lr is not None:
        config["optimizer"]["lr"] = float(lr)
    config["geometry"]["enabled"] = True
    config["geometry"]["cadence_steps"] = 1
    config["geometry"]["include_radius_angle"] = False
    config["run"].update({
        "id": run_id,
        "dir": run_dir.relative_to(ROOT).as_posix(),
        "save_every_steps": 1,
        "log_every_steps": 1,
        "keep_last_checkpoints": 8,
    })
    return config


def seed_cache(source: Path | None, run_dir: Path) -> None:
    if source is None:
        return
    destination = run_dir / "cache"
    destination.mkdir(parents=True, exist_ok=False)
    for item in source.iterdir():
        if not item.is_file():
            continue
        target = destination / item.name
        try:
            os.link(item, target)
        except OSError:
            import shutil
            shutil.copy2(item, target)


def run(config: Path, run_dir: Path, steps: int) -> None:
    command = [sys.executable, str(RGE), "--config", str(config),
               "--out", str(run_dir), "--resume", "none",
               "--max-steps", str(steps), "--geometry", "config"]
    print("$ " + " ".join(command))
    process = subprocess.run(command, cwd=ROOT, env={**os.environ, **ENV})
    if process.returncode:
        raise SystemExit(f"calibration arm failed with exit code {process.returncode}")


def geometry_map(run_dir: Path, *, exclude_through_step: int) -> dict[tuple[int, str], float]:
    rows = [json.loads(line) for line in (run_dir / "geometry.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    result = {}
    for row in rows:
        step = int(row["step"])
        if step <= exclude_through_step:
            continue
        value = float(row["base_relative_effective_step"])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("invalid base-relative effective step")
        result[(step, str(row["module"]))] = value
    return result


def run_summary(run_dir: Path) -> dict:
    rows = [json.loads(line) for line in (run_dir / "train_log.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    losses = [float(row["loss"]) for row in rows]
    if not losses or not all(math.isfinite(value) for value in losses):
        raise ValueError("calibration loss sequence is empty or non-finite")
    return {
        "losses": losses,
        "final_loss": losses[-1],
        "supervised_tokens_seen": int(manifest["final_position"]["supervised_tokens_seen"]),
        "active_gpu_hours": float(manifest["active_gpu_hours"]),
        "peak_reserved_mib": float(manifest["peak_reserved_mib"]),
        "source_commit": manifest["git"]["commit"],
        "source_fingerprint": manifest["source_fingerprint"]["combined"],
        "plan_hash": manifest["plan_hash"],
        "first_batches_sha256": manifest["first_batches_sha256"],
        "mix_sha256": manifest["data"]["mix_sha256"],
        "initial_state_sha256": manifest["trainable_initial_state"]["combined_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", default="configs/rge_peft/m2_o_a_adamw.yaml")
    parser.add_argument("--candidate", default="configs/rge_peft/m2_o_m_muon.yaml")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--exclude-through-step", type=int, default=1)
    parser.add_argument("--muon-lrs", nargs="+", type=float,
                        default=[5.0e-5, 1.0e-4, 2.0e-4])
    parser.add_argument("--out", default="artifacts/rge_peft/calibration/m2_muon_lr_v1")
    parser.add_argument("--cache-from", default="artifacts/heal_runs/b0/cache")
    parser.add_argument("--evidence",
                        default="experiments/rge_peft/evidence/m2_muon_lr_calibration.json")
    args = parser.parse_args()
    if args.steps < 2 or not 0 <= args.exclude_through_step < args.steps:
        raise SystemExit("calibration requires measured steps after the exclusion boundary")
    lrs = sorted(set(float(value) for value in args.muon_lrs))
    if len(lrs) < 3 or any(not math.isfinite(value) or value <= 0.0 for value in lrs):
        raise SystemExit("provide at least three distinct positive finite Muon LRs")

    control_path, candidate_path = ROOT / args.control, ROOT / args.candidate
    control_source, candidate_source = load_yaml(control_path), load_yaml(candidate_path)
    comparable_control = copy.deepcopy(control_source)
    comparable_candidate = copy.deepcopy(candidate_source)
    comparable_control.pop("optimizer"); comparable_control.pop("run")
    comparable_candidate.pop("optimizer"); comparable_candidate.pop("run")
    if comparable_control != comparable_candidate:
        raise SystemExit("calibration source configs differ outside optimizer/run")
    pair = load_yaml(PAIR)
    if pair["execution_gate"]["status"] != "AUTHORIZED":
        raise SystemExit("M1 execution gate is not authorized")
    if pair["optimizer_calibration"]["status"] != "provisional_screening_configuration":
        raise SystemExit("calibration is not in the provisional preregistered state")

    output = ROOT / args.out
    if output.exists():
        raise SystemExit(f"calibration output already exists: {output}")
    output.mkdir(parents=True)
    config_dir = output / "configs"
    config_dir.mkdir()
    cache_source = ROOT / args.cache_from if args.cache_from else None
    if cache_source is not None and not cache_source.is_dir():
        raise SystemExit(f"cache source does not exist: {cache_source}")
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True).strip()
    preregistration = {
        "schema_version": 1,
        "status": "PREREGISTERED_BEFORE_RUNS",
        "git_commit": git_commit,
        "steps": args.steps,
        "exclude_through_step": args.exclude_through_step,
        "control_lr": float(control_source["optimizer"]["lr"]),
        "candidate_lrs": lrs,
        "candidate_extra_scale_factor": float(
            candidate_source["optimizer"]["extra_scale_factor"]),
        "target_ratios": list(TARGET_RATIOS),
        "target_ratio_factor_tolerance": TARGET_FACTOR_TOLERANCE,
        "selection_metric": "median paired module-step rho_delta ratio to AdamW",
        "selection_rule": "stable candidate closest to ratio 1.0 in absolute log2 distance",
        "evaluation_used": False,
        "sealed_data_accessed": False,
        "control_config_sha256": sha256_file(control_path),
        "candidate_config_sha256": sha256_file(candidate_path),
        "calibration_script_sha256": sha256_file(Path(__file__)),
        "pair_manifest_sha256": sha256_file(PAIR),
    }
    atomic_json(output / "calibration_preregistration.json", preregistration)

    arms = [("adamw", control_source, None)] + [
        (f"muon_lr_{lr_label(lr)}", candidate_source, lr) for lr in lrs]
    run_dirs = {}
    for name, source, lr in arms:
        run_dir = output / name
        run_dirs[name] = run_dir
        config = calibration_config(
            source, run_id=f"rge_v1__m2__calibration__{name}__seed20260819",
            run_dir=run_dir, lr=lr)
        config_path = config_dir / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False),
                               encoding="utf-8", newline="\n")
        seed_cache(cache_source, run_dir)
        run(config_path, run_dir, args.steps)

    control_geometry = geometry_map(
        run_dirs["adamw"], exclude_through_step=args.exclude_through_step)
    control_summary = run_summary(run_dirs["adamw"])
    candidates = []
    for lr in lrs:
        name = f"muon_lr_{lr_label(lr)}"
        candidate_geometry = geometry_map(
            run_dirs[name], exclude_through_step=args.exclude_through_step)
        if set(candidate_geometry) != set(control_geometry):
            raise SystemExit(f"geometry key mismatch for {name}")
        ratios = [candidate_geometry[key] / control_geometry[key]
                  for key in sorted(control_geometry) if control_geometry[key] > 0.0]
        if not ratios or not all(math.isfinite(value) for value in ratios):
            raise SystemExit(f"invalid effective-step ratios for {name}")
        summary = run_summary(run_dirs[name])
        for field in ("plan_hash", "first_batches_sha256", "mix_sha256",
                      "initial_state_sha256", "supervised_tokens_seen"):
            if summary[field] != control_summary[field]:
                raise SystemExit(f"calibration comparability mismatch {field} for {name}")
        candidates.append({
            "lr": lr, "name": name, "stable": True,
            "paired_values": len(ratios),
            "rho_ratio_median": statistics.median(ratios),
            "rho_ratio_p10": quantile(ratios, 0.10),
            "rho_ratio_p90": quantile(ratios, 0.90),
            **summary,
        })
    selected = min(candidates, key=lambda payload: abs(math.log2(
        payload["rho_ratio_median"])))
    regimes = {
        str(target): min(candidates, key=lambda payload: abs(math.log2(
            payload["rho_ratio_median"] / target)))["lr"]
        for target in TARGET_RATIOS
    }
    regime_payloads = {
        str(target): min(candidates, key=lambda payload: abs(math.log2(
            payload["rho_ratio_median"] / target)))
        for target in TARGET_RATIOS
    }
    coverage = all(
        1.0 / TARGET_FACTOR_TOLERANCE
        <= regime_payloads[str(target)]["rho_ratio_median"] / target
        <= TARGET_FACTOR_TOLERANCE
        for target in TARGET_RATIOS)
    result = {
        **preregistration,
        "status": "PASS" if coverage else "BRACKET_INSUFFICIENT",
        "control": control_summary,
        "candidates": candidates,
        "identified_regime_lrs": regimes,
        "selected_lr": selected["lr"],
        "selected_ratio": selected["rho_ratio_median"],
        "target_regime_coverage_passed": coverage,
        "selection_frozen_for_exploratory_pair": coverage,
        "calibration_output": output.relative_to(ROOT).as_posix(),
    }
    atomic_json(output / "calibration_summary.json", result)
    atomic_json(ROOT / args.evidence, result)
    print(json.dumps({
        "status": result["status"], "selected_lr": result["selected_lr"],
        "selected_ratio": result["selected_ratio"], "regimes": regimes,
        "candidates": [{key: item[key] for key in
                        ("lr", "rho_ratio_median", "rho_ratio_p10",
                         "rho_ratio_p90", "final_loss", "active_gpu_hours")}
                       for item in candidates],
        "sealed_data_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
