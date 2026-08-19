"""Cross-seed aggregation.

The protocol asks for at least three seeds. A single-seed difference is not a
result: LoRA runs vary, and if the seed-to-seed spread is comparable to the
effect, the honest verdict is "inconclusive at this power" rather than a
direction.

This module combines several per-seed comparison reports into one statement,
and refuses to hide the variance that motivated running multiple seeds.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rineya_lab.config import ComparisonConfig
from rineya_lab.utils.io import read_json, write_json
from rineya_lab.utils.logging import get_logger
from rineya_lab.version import __version__

LOGGER = get_logger("analysis.multiseed")

# Two-sided Student t critical values at alpha=0.05, keyed by DEGREES OF FREEDOM
# (= seeds - 1). Seed counts in this kind of study are tiny, so a normal
# approximation would understate the interval.
_T_CRITICAL_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
                  7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179}


@dataclass
class SeedEntry:
    seed: Any
    path: str
    difference_pp: float
    ci_low_pp: float
    ci_high_pp: float
    baseline_mean: float
    candidate_mean: float
    n_paired: int
    decision: dict[str, Any] = field(default_factory=dict)


def _t_critical(degrees_of_freedom: int) -> float | None:
    """Two-sided 95% t critical value, or None when there is no variance estimate."""
    if degrees_of_freedom < 1:
        return None
    return _T_CRITICAL_95.get(degrees_of_freedom, 1.96)


def collect_seed_entries(report_paths: list[str | Path], comparison_key: str | None = None) -> dict[str, list[SeedEntry]]:
    """Load per-seed comparison reports and group entries by comparison key."""
    grouped: dict[str, list[SeedEntry]] = {}
    for path in report_paths:
        path = Path(path)
        payload = read_json(path / "comparison_report.json" if path.is_dir() else path)
        seed = payload.get("settings", {}).get("seed")
        candidate_seed = payload.get("candidate", {}).get("name")
        for key, comparison in payload.get("comparisons", {}).items():
            if comparison_key is not None and key != comparison_key:
                continue
            bootstrap = comparison["bootstrap"]
            grouped.setdefault(key, []).append(
                SeedEntry(
                    seed=seed if seed is not None else candidate_seed,
                    path=str(path),
                    difference_pp=bootstrap["observed_difference_pp"],
                    ci_low_pp=bootstrap["ci_low_pp"],
                    ci_high_pp=bootstrap["ci_high_pp"],
                    baseline_mean=bootstrap["baseline_mean"],
                    candidate_mean=bootstrap["candidate_mean"],
                    n_paired=comparison["n_paired"],
                    decision=comparison.get("decision", {}),
                )
            )
    return grouped


def aggregate(entries: list[SeedEntry], cfg: ComparisonConfig) -> dict[str, Any]:
    """Combine per-seed differences into a seed-level interval and verdict."""
    differences = [entry.difference_pp for entry in entries]
    seeds = len(differences)
    mean_difference = statistics.fmean(differences)
    between_seed_sd = statistics.stdev(differences) if seeds > 1 else 0.0
    standard_error = between_seed_sd / math.sqrt(seeds) if seeds > 1 else 0.0
    critical = _t_critical(seeds - 1)

    if critical is None:
        interval = None
        note = "A single seed provides no estimate of run-to-run variance. Run at least three."
    else:
        interval = [
            round(mean_difference - critical * standard_error, 4),
            round(mean_difference + critical * standard_error, 4),
        ]
        note = f"Seed-level t interval on {seeds} seeds (df={seeds - 1}, t={critical})."

    margin = cfg.non_inferiority_margin_pp
    verdict, rationale = "not_established", []
    if interval is None:
        verdict = "insufficient_seeds"
        rationale.append(note)
    else:
        rationale.append(f"mean difference {mean_difference:+.3f} pp, seed-level CI {interval} pp")
        if between_seed_sd > margin:
            verdict = "inconclusive_at_this_power"
            rationale.append(
                f"between-seed SD {between_seed_sd:.3f} pp exceeds the {margin} pp margin: "
                "run-to-run variation is larger than the effect the margin is meant to resolve. "
                "Report this as inconclusive rather than as a direction."
            )
        elif interval[0] > 0:
            verdict = "candidate_superior"
        elif interval[1] < 0:
            verdict = "baseline_superior"
        elif interval[0] > -margin:
            verdict = "non_inferior"
            rationale.append(f"CI lower bound {interval[0]:+.3f} pp clears the -{margin} pp margin")
        else:
            rationale.append(f"CI lower bound {interval[0]:+.3f} pp does not clear the -{margin} pp margin")

    directions = {"positive" if d > 0 else "negative" if d < 0 else "zero" for d in differences}
    if len(directions - {"zero"}) > 1:
        rationale.append("WARNING: seeds disagree in sign; the effect is not stable across runs.")

    if not cfg.margin_frozen:
        rationale.append(
            "WARNING: comparison.margin_frozen is false. Freeze and justify the margin in "
            "docs/PREREGISTRATION.md before treating any verdict as confirmatory."
        )

    return {
        "seeds": seeds,
        "seed_ids": [entry.seed for entry in entries],
        "per_seed_difference_pp": differences,
        "per_seed_baseline_mean": [entry.baseline_mean for entry in entries],
        "per_seed_candidate_mean": [entry.candidate_mean for entry in entries],
        "per_seed_n_paired": [entry.n_paired for entry in entries],
        "mean_difference_pp": round(mean_difference, 4),
        "median_difference_pp": round(statistics.median(differences), 4),
        "min_difference_pp": round(min(differences), 4),
        "max_difference_pp": round(max(differences), 4),
        "between_seed_sd_pp": round(between_seed_sd, 4),
        "standard_error_pp": round(standard_error, 4),
        "seed_level_ci_pp": interval,
        "non_inferiority_margin_pp": margin,
        "margin_frozen": cfg.margin_frozen,
        "verdict": verdict,
        "rationale": rationale,
        "note": note,
    }


def aggregate_reports(
    report_paths: list[str | Path],
    cfg: ComparisonConfig,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    grouped = collect_seed_entries(report_paths)
    if not grouped:
        raise ValueError("No comparisons found in the supplied reports")
    payload: dict[str, Any] = {
        "lab_version": __version__,
        "inputs": [str(path) for path in report_paths],
        "settings": {
            "non_inferiority_margin_pp": cfg.non_inferiority_margin_pp,
            "margin_frozen": cfg.margin_frozen,
            "alpha": cfg.alpha,
        },
        "comparisons": {key: aggregate(entries, cfg) for key, entries in sorted(grouped.items())},
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        write_json(output_dir / "multiseed_report.json", payload)
        (output_dir / "multiseed_report.md").write_text(render_markdown(payload), encoding="utf-8")
        LOGGER.info("Wrote %s", output_dir / "multiseed_report.json")
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Cross-seed aggregation",
        "",
        f"- inputs: {len(payload['inputs'])} comparison report(s)",
        f"- non-inferiority margin: {payload['settings']['non_inferiority_margin_pp']} pp "
        f"({'frozen' if payload['settings']['margin_frozen'] else 'PLACEHOLDER — NOT FROZEN'})",
        "",
        "| comparison | seeds | mean diff (pp) | between-seed SD | seed-level CI (pp) | verdict |",
        "|---|---:|---:|---:|---|---|",
    ]
    for key, summary in payload["comparisons"].items():
        interval = summary["seed_level_ci_pp"]
        interval_text = "n/a" if interval is None else f"[{interval[0]:+.2f}, {interval[1]:+.2f}]"
        lines.append(
            f"| `{key}` | {summary['seeds']} | {summary['mean_difference_pp']:+.2f} | "
            f"{summary['between_seed_sd_pp']:.2f} | {interval_text} | {summary['verdict']} |"
        )
    for key, summary in payload["comparisons"].items():
        lines += ["", f"### {key}", "", f"- per-seed differences (pp): {summary['per_seed_difference_pp']}"]
        lines += [f"- {line}" for line in summary["rationale"]]
    lines.append("")
    return "\n".join(lines)
