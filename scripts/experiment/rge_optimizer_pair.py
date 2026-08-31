#!/usr/bin/env python3
"""Run and audit the matched AdamW-LoRA versus Muon-LoRA RGE experiment.

The pair manifest is a preflight contract, not a convenience launcher. It
refuses differences in model, data, seed, LoRA capacity, schedule, curriculum,
geometry, memory policy, or supervised-token budget. Only the optimizer and
arm-specific run identity may differ. Arms execute sequentially so a single
consumer GPU is never oversubscribed.
"""

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
import time
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
RGE = ROOT / "scripts" / "train" / "rge_peft.py"
EXPORT = ROOT / "scripts" / "train" / "export_adapter.py"
EVALUATE = ROOT / "scripts" / "eval" / "run_gates.py"
PAIR_SCHEMA = "rge-optimizer-pair-v1"
ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYTHONIOENCODING": "utf-8",
}
RETENTION_METRICS = {
    "mean_quality_score": "higher",
    "quality_pass_rate": "higher",
    "intelligibility_pass_rate": "higher",
    "critical_defect_case_rate": "lower",
    "contract_violation_case_rate": "lower",
    "mean_contract_fulfillment_score": "higher",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a mapping in {path}")
    return payload


def _without_arm_specific(config: dict) -> dict:
    comparable = copy.deepcopy(config)
    comparable.pop("optimizer", None)
    comparable.pop("run", None)
    return comparable


def _contains_sealed(value: Any) -> bool:
    if isinstance(value, str):
        return "SEALED" in value.upper()
    if isinstance(value, dict):
        return any(_contains_sealed(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_sealed(item) for item in value)
    return False


def validate_pair_manifest(pair_path: Path) -> tuple[dict, dict[str, dict], list[str]]:
    """Validate the registered scientific substrate without loading a model."""
    pair = load_yaml(pair_path)
    problems: list[str] = []
    if pair.get("schema_version") != PAIR_SCHEMA:
        problems.append(f"schema_version must be {PAIR_SCHEMA}")
    for key in (
        "experiment_id",
        "execution_gate",
        "arms",
        "fixed_budget",
        "dataset_roles",
        "evaluation",
        "claim_boundary",
        "output_dir",
    ):
        if key not in pair:
            problems.append(f"pair manifest missing {key}")

    gate = pair.get("execution_gate", {})
    if gate.get("status") not in {"BLOCKED", "AUTHORIZED"}:
        problems.append("execution_gate.status must be BLOCKED or AUTHORIZED")
    if not gate.get("required_milestone"):
        problems.append("execution_gate.required_milestone must be explicit")
    calibration = pair.get("optimizer_calibration", {})
    if calibration.get("status") not in {
            "provisional_screening_configuration", "frozen_exploratory"}:
        problems.append("optimizer_calibration.status is invalid")
    policy = pair.get("execution_policy", {})
    if policy.get("hardware_class") != "single_consumer_gpu":
        problems.append("execution_policy.hardware_class must be single_consumer_gpu")
    if policy.get("concurrency") != "sequential_only":
        problems.append("execution_policy.concurrency must be sequential_only")
    if policy.get("default_order") not in {
        "control-candidate", "candidate-control"
    }:
        problems.append("execution_policy.default_order is invalid")
    if policy.get("reverse_order_repeat") != (
        "required_before_any_confirmatory_hardware_speed_claim"
    ):
        problems.append("execution_policy must require a reverse-order speed replicate")

    arms = pair.get("arms", {})
    if set(arms) != {"control", "candidate"}:
        problems.append("arms must contain exactly control and candidate")

    configs: dict[str, dict] = {}
    for arm in ("control", "candidate"):
        if arm not in arms or "config" not in arms.get(arm, {}):
            continue
        path = repo_path(arms[arm]["config"])
        if not path.is_file():
            problems.append(f"{arm} config does not exist: {path}")
            continue
        configs[arm] = load_yaml(path)

    if len(configs) == 2:
        control, candidate = configs["control"], configs["candidate"]
        if _without_arm_specific(control) != _without_arm_specific(candidate):
            problems.append(
                "control/candidate differ outside optimizer and run sections; "
                "the pair is not capacity/data/schedule matched"
            )
        if str(control.get("optimizer", {}).get("name", "")).lower() != "adamw":
            problems.append("control optimizer must be adamw")
        if str(candidate.get("optimizer", {}).get("name", "")).lower() != "muon":
            problems.append("candidate optimizer must be muon")
        if control.get("run", {}).get("dir") == candidate.get("run", {}).get("dir"):
            problems.append("arm run directories must be distinct")
        if control.get("run", {}).get("id") == candidate.get("run", {}).get("id"):
            problems.append("arm run IDs must be distinct")

        fixed = pair.get("fixed_budget", {})
        requested = fixed.get("requested_supervised_tokens")
        if fixed.get("unit") != "supervised_tokens":
            problems.append("fixed_budget.unit must be supervised_tokens")
        budgets = {
            control.get("train", {}).get("supervised_token_budget"),
            candidate.get("train", {}).get("supervised_token_budget"),
        }
        if requested is None or int(requested) <= 0:
            problems.append("fixed_budget.requested_supervised_tokens must be positive")
        elif budgets != {int(requested)}:
            problems.append(
                f"arm budgets {budgets} do not equal requested budget {requested}"
            )

        learning = pair.get("dataset_roles", {}).get("learning", {}).get("path")
        arm_mix = control.get("data", {}).get("mix")
        if learning != arm_mix:
            problems.append(
                f"dataset_roles.learning.path={learning!r} does not match "
                f"arm data.mix={arm_mix!r}"
            )
        if _contains_sealed(control.get("data", {})) or _contains_sealed(
            candidate.get("data", {})
        ):
            problems.append("a training config references a sealed asset")

    active_eval = [
        item
        for item in pair.get("evaluation", {}).get("datasets", [])
        if item.get("enabled") is True
    ]
    if _contains_sealed(active_eval):
        problems.append("active evaluation datasets may not reference SEALED assets")
    retention_assets = {
        str(item.get("name")): item
        for item in pair.get("dataset_roles", {})
        .get("retention", {})
        .get("assets", [])
    }
    for dataset in active_eval:
        name = str(dataset.get("name"))
        asset = retention_assets.get(name)
        if asset is None:
            problems.append(f"active evaluation dataset {name!r} is not frozen as retention")
        elif not asset.get("sha256"):
            problems.append(f"retention asset {name!r} has no sha256")

    sealed = pair.get("dataset_roles", {}).get("sealed", {})
    if sealed.get("access") != "forbidden_until_m7":
        problems.append("dataset_roles.sealed.access must be forbidden_until_m7")
    capability = pair.get("dataset_roles", {}).get("capability", {})
    claim_mode = pair.get("claim_boundary", {}).get("maximum_claim")
    if capability.get("status") != "frozen" and claim_mode != "mechanism_and_retention_only":
        problems.append(
            "an unfrozen capability bank requires "
            "maximum_claim=mechanism_and_retention_only"
        )

    if len(configs) == 2:
        output = repo_path(pair.get("output_dir", ""))
        run_dirs = {
            repo_path(configs[arm].get("run", {}).get("dir", ""))
            for arm in ("control", "candidate")
        }
        if output in run_dirs:
            problems.append("pair output_dir must not equal an arm run directory")

    return pair, configs, problems


def run_process(command: list[str], *, label: str) -> None:
    print("$ " + " ".join(command))
    process = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, **ENV},
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {process.returncode}")


def run_arm(
    config_path: Path,
    *,
    resume: str,
    max_steps: int | None,
    out: Path | None,
) -> None:
    command = [
        sys.executable,
        str(RGE),
        "--config",
        str(config_path),
        "--resume",
        resume,
    ]
    if max_steps is not None:
        command += ["--max-steps", str(max_steps)]
    if out is not None:
        command += ["--out", str(out)]
    run_process(command, label=config_path.stem)


def load_rows(run_dir: Path) -> list[dict]:
    path = run_dir / "train_log.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing training log: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for index, row in enumerate(rows):
        for key in ("loss", "lr", "step_time_s"):
            value = float(row.get(key, 0.0))
            if not math.isfinite(value):
                raise ValueError(f"non-finite {key} in {path} row {index + 1}")
    return rows


def tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def latest_checkpoint(run_dir: Path) -> Path:
    checkpoint_dir = run_dir / "checkpoints"
    pointer = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    return checkpoint_dir / pointer["checkpoint"]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[position])


def timing_distribution(values: list[float], *, warmup_steps: int = 5) -> dict:
    steady = values[min(warmup_steps, len(values)) :]

    def payload(items: list[float]) -> dict:
        return {
            "count": len(items),
            "mean_s": statistics.fmean(items) if items else None,
            "median_s": statistics.median(items) if items else None,
            "p90_s": percentile(items, 0.90),
        }

    return {
        "all_steps": payload(values),
        "steady_state": payload(steady),
        "steady_state_excludes_first_steps": min(warmup_steps, len(values)),
    }


def summarize_run(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = load_rows(run_dir)
    cumulative = 0
    weighted_loss = 0.0
    curve = []
    step_times: list[float] = []
    for row in rows:
        tokens = int(row["supervised_tokens"])
        cumulative += tokens
        weighted_loss += float(row["loss"]) * tokens
        step_seconds = float(row.get("step_time_s", 0.0))
        step_times.append(step_seconds)
        curve.append(
            {
                "step": int(row["step"]),
                "supervised_tokens_seen": cumulative,
                "loss": float(row["loss"]),
                "lr": float(row["lr"]),
                "step_time_s": step_seconds,
            }
        )
    checkpoint = torch.load(
        latest_checkpoint(run_dir), map_location="cpu", weights_only=False
    )
    final_position = manifest.get("final_position") or checkpoint.get("position") or {}
    actual_tokens = int(final_position.get("supervised_tokens_seen", cumulative))
    total_step_seconds = sum(step_times)
    source = manifest.get("source_fingerprint", {})
    initial = manifest.get("trainable_initial_state", {})
    environment = manifest.get("environment", {})
    return {
        "run_dir": display_path(run_dir),
        "run_id": manifest["run_id"],
        "program_version": manifest.get("program_version"),
        "optimizer": manifest["optimizer"],
        "environment": environment,
        "memory_contract": manifest.get("memory"),
        "trainable_parameter_count": initial.get("parameter_count"),
        "comparability": {
            "program_version": manifest.get("program_version"),
            "git_commit": manifest.get("git", {}).get("commit"),
            "source_fingerprint_sha256": source.get("combined"),
            "model_sha256": manifest["model"]["combined_sha256"],
            "tokenizer_sha256": manifest["tokenizer"]["sha256"],
            "mix_sha256": manifest["data"]["mix_sha256"],
            "evaluation_assets_manifest_sha256": manifest.get(
                "evaluation_assets", {}
            ).get("manifest_sha256"),
            "seed": manifest["seed"],
            "environment": environment,
            "precision": manifest["precision"],
            "lora": manifest["lora"],
            "geometry": manifest["geometry"],
            "schedule": manifest["schedule"],
            "curriculum": manifest["curriculum"],
            "cma": manifest["cma"],
            "memory": manifest.get("memory"),
            "requested_supervised_token_budget": manifest[
                "supervised_token_budget"
            ],
            "plan_hash": manifest["plan_hash"],
            "first_batches_sha256": manifest["first_batches_sha256"],
            "trainable_initial_state_sha256": initial["combined_sha256"],
            "trainable_parameter_count": initial.get("parameter_count"),
        },
        "actual_supervised_tokens": actual_tokens,
        "steps": len(rows),
        "initial_loss": curve[0]["loss"] if curve else None,
        "final_loss": curve[-1]["loss"] if curve else None,
        "minimum_loss": min((point["loss"] for point in curve), default=None),
        "token_weighted_mean_loss": weighted_loss / cumulative if cumulative else None,
        "step_time_s": total_step_seconds,
        "step_time_distribution": timing_distribution(step_times),
        "supervised_tokens_per_s": (
            cumulative / total_step_seconds if total_step_seconds else None
        ),
        "active_gpu_hours": manifest.get("active_gpu_hours"),
        "wall_time_s": manifest.get("wall_time_s"),
        "peak_allocated_mib": manifest.get("peak_allocated_mib"),
        "peak_reserved_mib": manifest.get("peak_reserved_mib"),
        "optimizer_state_tensor_bytes": tensor_bytes(
            checkpoint.get("optimizer_state", {})
        ),
        "exit_reason": manifest.get("exit_reason"),
        "curve": curve,
        "summary": summary,
    }


def compare_completed_runs(
    control: dict, candidate: dict, *, require_budget: bool = True
) -> dict:
    left, right = control["comparability"], candidate["comparability"]
    mismatches = {
        key: {"control": left.get(key), "candidate": right.get(key)}
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    }
    if control["actual_supervised_tokens"] != candidate["actual_supervised_tokens"]:
        mismatches["actual_supervised_tokens"] = {
            "control": control["actual_supervised_tokens"],
            "candidate": candidate["actual_supervised_tokens"],
        }
    requested = left.get("requested_supervised_token_budget")
    if require_budget and requested is not None:
        for arm, payload in (("control", control), ("candidate", candidate)):
            if int(payload["actual_supervised_tokens"]) < int(requested):
                mismatches[f"{arm}_budget_not_reached"] = {
                    "requested": int(requested),
                    "actual": int(payload["actual_supervised_tokens"]),
                    "exit_reason": payload.get("exit_reason"),
                }
    control_axis = [point["supervised_tokens_seen"] for point in control.get("curve", [])]
    candidate_axis = [
        point["supervised_tokens_seen"] for point in candidate.get("curve", [])
    ]
    if control_axis != candidate_axis:
        mismatches["learning_curve_token_axis"] = {
            "control": control_axis,
            "candidate": candidate_axis,
        }

    pointwise = []
    if control_axis == candidate_axis:
        for left_point, right_point in zip(control.get("curve", []), candidate.get("curve", [])):
            pointwise.append(
                {
                    "supervised_tokens_seen": left_point["supervised_tokens_seen"],
                    "control_loss": left_point["loss"],
                    "candidate_loss": right_point["loss"],
                    "candidate_minus_control": right_point["loss"] - left_point["loss"],
                }
            )

    control_steady = control["step_time_distribution"]["steady_state"]
    candidate_steady = candidate["step_time_distribution"]["steady_state"]
    metrics = {
        "final_loss_delta_candidate_minus_control": (
            candidate["final_loss"] - control["final_loss"]
            if control["final_loss"] is not None and candidate["final_loss"] is not None
            else None
        ),
        "weighted_loss_delta_candidate_minus_control": (
            candidate["token_weighted_mean_loss"]
            - control["token_weighted_mean_loss"]
            if control["token_weighted_mean_loss"] is not None
            and candidate["token_weighted_mean_loss"] is not None
            else None
        ),
        "training_time_ratio_candidate_over_control": (
            candidate["step_time_s"] / control["step_time_s"]
            if control["step_time_s"]
            else None
        ),
        "throughput_ratio_candidate_over_control": (
            candidate["supervised_tokens_per_s"]
            / control["supervised_tokens_per_s"]
            if control["supervised_tokens_per_s"]
            else None
        ),
        "steady_median_step_time_ratio_candidate_over_control": (
            candidate_steady["median_s"] / control_steady["median_s"]
            if control_steady.get("median_s") and candidate_steady.get("median_s")
            else None
        ),
        "optimizer_state_bytes_ratio_candidate_over_control": (
            candidate["optimizer_state_tensor_bytes"]
            / control["optimizer_state_tensor_bytes"]
            if control["optimizer_state_tensor_bytes"]
            else None
        ),
        "peak_reserved_mib_delta_candidate_minus_control": (
            candidate["peak_reserved_mib"] - control["peak_reserved_mib"]
            if candidate.get("peak_reserved_mib") is not None
            and control.get("peak_reserved_mib") is not None
            else None
        ),
    }
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "mismatches": mismatches,
        "screening_metrics": metrics,
        "pointwise_loss_curve": pointwise,
    }


def retention_asset_map(pair: dict) -> dict[str, dict]:
    return {
        str(item["name"]): item
        for item in pair["dataset_roles"]["retention"].get("assets", [])
    }


def evaluate_final(
    pair: dict,
    configs: dict[str, dict],
    run_dirs: dict[str, Path],
    output: Path,
) -> dict:
    results = {}
    frozen = retention_asset_map(pair)
    for arm in ("control", "candidate"):
        adapter = output / "adapters" / arm
        run_process(
            [
                sys.executable,
                str(EXPORT),
                str(run_dirs[arm]),
                "--out",
                str(adapter),
            ],
            label=f"export-{arm}",
        )
        results[arm] = {}
        for dataset in pair["evaluation"].get("datasets", []):
            if dataset.get("enabled") is not True:
                continue
            name = str(dataset["name"])
            if "SEALED" in name.upper():
                raise SystemExit("refusing sealed dataset in active evaluation")
            destination = output / "evaluation" / arm / name
            command = [
                sys.executable,
                str(EVALUATE),
                "--dataset",
                name,
                "--model",
                str(configs[arm]["model"]["path"]),
                "--adapter",
                str(adapter),
                "--out",
                str(destination),
            ]
            limit = dataset.get("limit")
            if limit is not None:
                command += ["--limit", str(int(limit))]
            run_process(command, label=f"evaluate-{arm}-{name}")
            evaluation_manifest = json.loads(
                (destination / "evaluation_manifest.json").read_text(encoding="utf-8")
            )
            expected_sha = frozen[name]["sha256"]
            observed_sha = evaluation_manifest.get("dataset_sha256")
            if observed_sha != expected_sha:
                raise SystemExit(
                    f"{arm}/{name} evaluated unexpected dataset hash: "
                    f"{observed_sha!r} != {expected_sha!r}"
                )
            results[arm][name] = {
                "summary": json.loads(
                    (destination / "summary.json").read_text(encoding="utf-8")
                ),
                "evaluation_manifest": evaluation_manifest,
            }
    return results


def compare_retention(evaluation: dict | None) -> dict | None:
    if not evaluation or set(evaluation) != {"control", "candidate"}:
        return None
    datasets = {}
    names = sorted(set(evaluation["control"]) & set(evaluation["candidate"]))
    for name in names:
        control = evaluation["control"][name]["summary"]
        candidate = evaluation["candidate"][name]["summary"]
        metrics = {}
        for metric, direction in RETENTION_METRICS.items():
            if metric not in control or metric not in candidate:
                continue
            left = float(control[metric])
            right = float(candidate[metric])
            raw_delta = right - left
            oriented = raw_delta if direction == "higher" else -raw_delta
            metrics[metric] = {
                "direction": direction,
                "control": left,
                "candidate": right,
                "candidate_minus_control": raw_delta,
                "oriented_improvement": oriented,
            }
        datasets[name] = {"metrics": metrics}
    return {
        "status": "MEASURED_NOT_PROMOTION_DECIDED",
        "datasets": datasets,
        "note": (
            "Named frozen retention comparison only; thresholds and broader "
            "capability claims remain governed by later phases."
        ),
    }


def parse_order(value: str) -> tuple[str, str]:
    mapping = {
        "control-candidate": ("control", "candidate"),
        "candidate-control": ("candidate", "control"),
    }
    return mapping[value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True)
    parser.add_argument(
        "--arm",
        choices=("none", "control", "candidate", "both"),
        default="none",
    )
    parser.add_argument("--resume", choices=("none", "auto"), default="none")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="bounded integration smoke; not confirmatory evidence",
    )
    parser.add_argument("--evaluate-final", action="store_true")
    parser.add_argument(
        "--order",
        choices=("control-candidate", "candidate-control"),
        default="control-candidate",
        help="order for --arm both; reverse-order repeats require separate run roots",
    )
    parser.add_argument(
        "--run-root",
        default=None,
        help="optional disposable root with control/ and candidate/ run directories",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="optional pair-artifact output override",
    )
    args = parser.parse_args()

    pair_path = repo_path(args.pair)
    pair, configs, problems = validate_pair_manifest(pair_path)
    output = repo_path(args.output or pair.get("output_dir", "artifacts/rge_peft/pairs/invalid"))
    output.mkdir(parents=True, exist_ok=True)
    config_hashes = {
        arm: sha256_file(repo_path(pair["arms"][arm]["config"]))
        for arm in ("control", "candidate")
        if arm in pair.get("arms", {})
        and repo_path(pair["arms"][arm]["config"]).is_file()
    }
    compact = {
        "schema_version": 1,
        "status": "FAIL" if problems else "PASS",
        "experiment_id": pair.get("experiment_id"),
        "pair_manifest_sha256": sha256_file(pair_path),
        "launcher_sha256": sha256_file(Path(__file__)),
        "config_sha256": config_hashes,
        "execution_gate": pair.get("execution_gate"),
        "requested_supervised_tokens": pair.get("fixed_budget", {}).get(
            "requested_supervised_tokens"
        ),
        "capability_bank_status": pair.get("dataset_roles", {})
        .get("capability", {})
        .get("status"),
        "maximum_claim": pair.get("claim_boundary", {}).get("maximum_claim"),
        "problems": problems,
        "sealed_data_accessed": False,
    }
    atomic_json(output / "pair_preflight.json", compact)
    print(json.dumps(compact, indent=2))
    if problems:
        raise SystemExit("pair preflight failed")
    if args.arm == "none":
        return
    if pair["execution_gate"].get("status") != "AUTHORIZED":
        raise SystemExit(
            "execution gate is not AUTHORIZED; validation succeeded, but training "
            f"remains blocked by {pair['execution_gate'].get('required_milestone')}"
        )
    if args.max_steps is None and pair.get("optimizer_calibration", {}).get(
            "status") != "frozen_exploratory":
        raise SystemExit(
            "optimizer calibration is not frozen; only bounded --max-steps "
            "integration smoke is authorized before the effective-step bracket"
        )
    if args.max_steps is not None and args.max_steps < 1:
        raise SystemExit("--max-steps must be positive")

    run_root = repo_path(args.run_root) if args.run_root else None
    run_dirs = {
        arm: (
            run_root / arm
            if run_root is not None
            else repo_path(configs[arm]["run"]["dir"])
        )
        for arm in ("control", "candidate")
    }
    if run_root is not None and run_dirs["control"] == run_dirs["candidate"]:
        raise SystemExit("run-root produced identical arm directories")

    order = parse_order(args.order)
    selected = order if args.arm == "both" else (args.arm,)
    execution = {
        "schema_version": 1,
        "experiment_id": pair["experiment_id"],
        "pair_manifest": display_path(pair_path),
        "pair_manifest_sha256": compact["pair_manifest_sha256"],
        "launcher_sha256": compact["launcher_sha256"],
        "config_sha256": config_hashes,
        "selected_arms": list(selected),
        "order": list(order),
        "resume": args.resume,
        "max_steps": args.max_steps,
        "bounded_smoke": args.max_steps is not None,
        "run_dirs": {arm: display_path(path) for arm, path in run_dirs.items()},
        "output": display_path(output),
        "started_unix_s": time.time(),
        "torch": torch.__version__,
        "sealed_data_accessed": False,
    }
    atomic_json(output / "pair_execution.json", execution)

    started = time.time()
    for arm in selected:
        run_arm(
            repo_path(pair["arms"][arm]["config"]),
            resume=args.resume,
            max_steps=args.max_steps,
            out=run_dirs[arm] if run_root is not None else None,
        )

    available = {}
    for arm in ("control", "candidate"):
        if (run_dirs[arm] / "summary.json").is_file():
            available[arm] = summarize_run(run_dirs[arm])
    comparison = None
    if set(available) == {"control", "candidate"}:
        comparison = compare_completed_runs(
            available["control"], available["candidate"],
            require_budget=args.max_steps is None,
        )
        if comparison["status"] != "PASS":
            atomic_json(
                output / "pair_summary.json",
                {
                    "status": "FAIL_COMPARABILITY",
                    "arms": available,
                    "comparison": comparison,
                    "sealed_data_accessed": False,
                },
            )
            raise SystemExit("completed arms are not scientifically matched")

    if args.evaluate_final and set(available) != {"control", "candidate"}:
        raise SystemExit("--evaluate-final requires both completed matched arms")
    evaluation = (
        evaluate_final(pair, configs, run_dirs, output) if args.evaluate_final else None
    )
    retention = compare_retention(evaluation)
    result = {
        "schema_version": 1,
        "status": "COMPLETE" if comparison else "PARTIAL",
        "experiment_id": pair["experiment_id"],
        "pair_manifest_sha256": compact["pair_manifest_sha256"],
        "launcher_sha256": compact["launcher_sha256"],
        "config_sha256": config_hashes,
        "bounded_smoke": args.max_steps is not None,
        "execution_order": list(selected),
        "elapsed_s": round(time.time() - started, 3),
        "arms": available,
        "comparison": comparison,
        "evaluation": evaluation,
        "retention_comparison": retention,
        "claim_boundary": pair["claim_boundary"],
        "interpretation": (
            "mechanism_and_retention_only; no intelligence or general capability claim"
            if pair["dataset_roles"]["capability"].get("status") != "frozen"
            else (
                "capability evidence may be interpreted only under the frozen "
                "evaluation protocol"
            )
        ),
        "sealed_data_accessed": False,
    }
    atomic_json(output / "pair_summary.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "arms"}, indent=2))


if __name__ == "__main__":
    main()
