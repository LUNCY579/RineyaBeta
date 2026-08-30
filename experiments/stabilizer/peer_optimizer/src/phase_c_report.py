#!/usr/bin/env python3
"""Decision aggregation and evidence-package helpers for PCSM Phase C."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence


_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
    19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_ci(values: Sequence[float], confidence: float = 0.95) -> dict[str, float]:
    if confidence != 0.95:
        raise ValueError("Phase-C v1 pre-registers a 95% paired t interval")
    n = len(values)
    if n < 2:
        return {"n": n, "mean": float(values[0]) if values else math.nan, "low": math.nan, "high": math.nan}
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    df = n - 1
    tcrit = _T95.get(df, 1.96)
    half = tcrit * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "low": mean - half, "high": mean + half}


def _arm_map(trials: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Mapping[str, Any]]]:
    by_seed: dict[int, dict[str, Mapping[str, Any]]] = {}
    for trial in trials:
        seed = int(trial["seed"])
        arm = str(trial["arm"])
        if arm in by_seed.setdefault(seed, {}):
            raise ValueError(f"duplicate trial for seed={seed}, arm={arm}")
        by_seed[seed][arm] = trial
    return by_seed


def summarize(
    trials: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    required_arms: Sequence[str],
    real_arm: str,
    shuffled_arm: str,
    layer_control_arm: str,
    min_win_fraction: float,
) -> dict[str, Any]:
    by_seed = _arm_map(trials)
    missing: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for seed in seeds:
        arms = by_seed.get(int(seed), {})
        for arm in required_arms:
            trial = arms.get(arm)
            if trial is None:
                missing.append({"seed": int(seed), "arm": arm})
            elif not bool(trial.get("valid", False)):
                invalid.append({"seed": int(seed), "arm": arm, "reasons": trial.get("invalid_reasons", [])})

    if missing:
        status = "INSUFFICIENT_EVIDENCE"
    elif invalid:
        status = "INVALID"
    else:
        status = "PENDING"

    coord_diffs: list[float] = []
    layer_diffs: list[float] = []
    coord_wins = 0
    layer_wins = 0
    paired_rows: list[dict[str, Any]] = []
    if status == "PENDING":
        for seed in seeds:
            arms = by_seed[int(seed)]
            real = arms[real_arm]
            shuffled = arms[shuffled_arm]
            layer = arms[layer_control_arm]
            # Lower post-update held-out loss is better.
            d_coord = float(shuffled["post_dev_loss"]) - float(real["post_dev_loss"])
            d_layer = float(layer["post_dev_loss"]) - float(real["post_dev_loss"])
            coord_diffs.append(d_coord)
            layer_diffs.append(d_layer)
            coord_wins += int(d_coord > 0)
            layer_wins += int(d_layer > 0)
            paired_rows.append(
                {
                    "seed": int(seed),
                    "real_post_dev_loss": float(real["post_dev_loss"]),
                    "shuffled_post_dev_loss": float(shuffled["post_dev_loss"]),
                    "layer_control_post_dev_loss": float(layer["post_dev_loss"]),
                    "shuffled_minus_real": d_coord,
                    "layer_control_minus_real": d_layer,
                }
            )

        coord_ci = paired_ci(coord_diffs)
        layer_ci = paired_ci(layer_diffs)
        min_wins = math.ceil(len(seeds) * min_win_fraction - 1e-12)
        gates = {
            "paired_seed_count": len(seeds),
            "required_seed_count": len(seeds),
            "coordinate_wins": coord_wins,
            "coordinate_required_wins": min_wins,
            "coordinate_ci_low_gt_zero": bool(coord_ci["low"] > 0),
            "layer_identity_wins": layer_wins,
            "layer_identity_required_wins": min_wins,
            "layer_identity_ci_low_gt_zero": bool(layer_ci["low"] > 0),
        }
        advance = (
            coord_wins >= min_wins
            and coord_ci["low"] > 0
            and layer_wins >= min_wins
            and layer_ci["low"] > 0
        )
        status = "ADVANCE_TO_MATCHED_SFT" if advance else "DO_NOT_ADVANCE"
    else:
        coord_ci = {}
        layer_ci = {}
        gates = {}

    arm_aggregates: dict[str, Any] = {}
    for arm in required_arms:
        rows = [trial for trial in trials if trial.get("arm") == arm and trial.get("valid")]
        if not rows:
            continue
        pre = [float(row["pre_dev_loss"]) for row in rows]
        post = [float(row["post_dev_loss"]) for row in rows]
        improvements = [a - b for a, b in zip(pre, post)]
        arm_aggregates[arm] = {
            "n": len(rows),
            "pre_dev_loss_mean": statistics.fmean(pre),
            "post_dev_loss_mean": statistics.fmean(post),
            "improvement_mean": statistics.fmean(improvements),
            "improvement_median": statistics.median(improvements),
        }

    return {
        "status": status,
        "missing_trials": missing,
        "invalid_trials": invalid,
        "coordinate_comparison": {"paired": paired_rows, "ci95": coord_ci},
        "layer_identity_comparison": {"ci95": layer_ci},
        "gates": gates,
        "arm_aggregates": arm_aggregates,
        "claim_boundary": (
            "A passing Phase-C result only justifies a matched full-SFT follow-up; "
            "it is not itself a Rineya quality or speed claim."
        ),
    }


def write_decision_markdown(summary: Mapping[str, Any], path: str | Path) -> None:
    lines = [
        "# PCSM Phase-C decision",
        "",
        f"**Status:** `{summary['status']}`",
        "",
        summary["claim_boundary"],
        "",
    ]
    coord = summary.get("coordinate_comparison", {}).get("ci95", {})
    layer = summary.get("layer_identity_comparison", {}).get("ci95", {})
    if coord:
        lines += [
            "## Primary causal comparison",
            "",
            f"- coordinate-shuffle minus real mean held-out loss: `{coord['mean']:+.8f}`",
            f"- 95% paired t CI: `[{coord['low']:+.8f}, {coord['high']:+.8f}]`",
        ]
    if layer:
        lines += [
            f"- layer-identity-control minus real mean held-out loss: `{layer['mean']:+.8f}`",
            f"- 95% paired t CI: `[{layer['low']:+.8f}, {layer['high']:+.8f}]`",
            "",
        ]
    lines += [
        "## Interpretation",
        "",
        "- `ADVANCE_TO_MATCHED_SFT`: causal screen passed; prepare a short, matched SFT pilot.",
        "- `DO_NOT_ADVANCE`: preserve the negative result; do not spend full-SFT compute yet.",
        "- `INVALID`: fix execution/provenance only and rerun without retuning the preregistered design.",
        "- `INSUFFICIENT_EVIDENCE`: complete missing paired seeds/arms.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def hash_evidence_tree(root: str | Path, *, exclude: Iterable[str] = ("ARTIFACT_HASHES.json",)) -> dict[str, str]:
    root_path = Path(root)
    excluded = set(exclude)
    hashes: dict[str, str] = {}
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        hashes[str(path.relative_to(root_path)).replace("\\", "/")] = sha256_file(path)
    return hashes
