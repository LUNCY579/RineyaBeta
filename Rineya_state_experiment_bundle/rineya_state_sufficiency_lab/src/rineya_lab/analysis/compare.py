"""Paired comparison of two evaluation runs.

Pairing is on ``problem_id``, never on the expanded transition row, and only
problems present in *both* runs are compared. Problems missing from one side are
reported explicitly rather than dropped silently, because an asymmetric
denominator is one of the easiest ways to manufacture a difference.

The primary end-to-end contrast is intentionally cross-protocol:

    FULL   evaluated by ``full_direct``       (its native inference mode)
    SPLIT  evaluated by ``generated_rollout`` (its native inference mode)

Comparing SPLIT's rollout against FULL's direct answer is the honest end-to-end
question: each arm is asked to solve the same held-out problem the way it was
trained to. Same-protocol pairs are also reported for the diagnostics.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rineya_lab.analysis.stats import (
    ComparisonDecision,
    decide,
    exact_mcnemar,
    holm_bonferroni,
    paired_bootstrap_ci,
    sign_flip_test,
)
from rineya_lab.config import ComparisonConfig, LabConfig
from rineya_lab.utils.hashing import stable_shuffle_key
from rineya_lab.utils.io import read_json, read_jsonl, write_json
from rineya_lab.utils.logging import get_logger
from rineya_lab.version import __version__

LOGGER = get_logger("analysis.compare")


@dataclass
class RunView:
    """One evaluation run loaded from disk."""

    path: Path
    report: dict[str, Any]
    records_by_protocol: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @property
    def condition(self) -> str:
        return self.report.get("experiment", {}).get("condition", "unknown")

    @property
    def name(self) -> str:
        return self.report.get("experiment", {}).get("name", self.path.name)

    @property
    def is_reference(self) -> bool:
        return bool(self.report.get("backend", {}).get("is_reference"))

    def protocols(self) -> list[str]:
        return sorted(self.records_by_protocol)

    def provenance(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "name": self.name,
            "condition": self.condition,
            "config_fingerprint": self.report.get("config_fingerprint"),
            "backend": self.report.get("backend", {}).get("backend"),
            "is_reference_backend": self.is_reference,
            "eval_split": self.report.get("eval_set", {}).get("split"),
            "eval_problem_ids_sha256": self.report.get("eval_set", {}).get("problem_ids_sha256"),
            "adapter_path": self.report.get("resolved_config", {})
            .get("evaluation", {})
            .get("backend", {})
            .get("adapter_path"),
        }


def load_run(run_dir: str | Path) -> RunView:
    run_dir = Path(run_dir)
    report_path = run_dir / "eval_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"No eval_report.json in {run_dir}; run `rineya-lab evaluate` first")
    view = RunView(path=run_dir, report=read_json(report_path))
    records_dir = run_dir / "records"
    if records_dir.exists():
        for path in sorted(records_dir.glob("*.jsonl")):
            view.records_by_protocol[path.stem] = read_jsonl(path)
    if not view.records_by_protocol:
        raise FileNotFoundError(
            f"{run_dir} has no per-example records. Re-run evaluation with "
            "evaluation.save_per_example_records=true; the paired tests need item-level data."
        )
    return view


def _per_problem_scores(records: list[dict[str, Any]], metric: str) -> dict[str, float]:
    """Collapse item-level records into one score per source problem."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for record in records:
        buckets[record["problem_id"]].append(1.0 if record.get(metric) else 0.0)
    return {problem_id: sum(values) / len(values) for problem_id, values in buckets.items()}


def _record_index(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        index[record["problem_id"]].append(record)
    return index


def _stratum_values(records: list[dict[str, Any]], key: str) -> dict[str, str]:
    """Map problem_id -> stratum label for a stratification key."""
    out: dict[str, str] = {}
    for record in records:
        value = record.get(key)
        if value is None and key == "failure_type":
            value = "unknown"
        out.setdefault(record["problem_id"], str(value))
    return out


@dataclass
class PairedComparison:
    label: str
    baseline_protocol: str
    candidate_protocol: str
    metric: str
    n_paired: int
    baseline_only: list[str]
    candidate_only: list[str]
    bootstrap: dict[str, Any]
    mcnemar: dict[str, Any]
    randomization: dict[str, Any]
    decision: dict[str, Any]
    strata: dict[str, Any] = field(default_factory=dict)
    track_disagreement: dict[str, Any] = field(default_factory=dict)
    audit_sample: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "baseline_protocol": self.baseline_protocol,
            "candidate_protocol": self.candidate_protocol,
            "metric": self.metric,
            "n_paired": self.n_paired,
            "baseline_only_problem_count": len(self.baseline_only),
            "candidate_only_problem_count": len(self.candidate_only),
            "baseline_only_examples": self.baseline_only[:10],
            "candidate_only_examples": self.candidate_only[:10],
            "bootstrap": self.bootstrap,
            "mcnemar": self.mcnemar,
            "randomization": self.randomization,
            "decision": self.decision,
            "strata": self.strata,
            "track_disagreement": self.track_disagreement,
            "audit_sample": self.audit_sample,
        }


def compare_protocol_pair(
    baseline: RunView,
    candidate: RunView,
    baseline_protocol: str,
    candidate_protocol: str,
    cfg: ComparisonConfig,
) -> PairedComparison | None:
    base_records = baseline.records_by_protocol.get(baseline_protocol)
    cand_records = candidate.records_by_protocol.get(candidate_protocol)
    if not base_records or not cand_records:
        LOGGER.info(
            "Skipping pair %s:%s (missing in %s)",
            baseline_protocol,
            candidate_protocol,
            "baseline" if not base_records else "candidate",
        )
        return None

    metric = cfg.primary_metric
    base_scores = _per_problem_scores(base_records, metric)
    cand_scores = _per_problem_scores(cand_records, metric)
    shared = sorted(set(base_scores) & set(cand_scores))
    if not shared:
        LOGGER.warning("No shared problems between %s and %s", baseline_protocol, candidate_protocol)
        return None

    baseline_values = [base_scores[pid] for pid in shared]
    candidate_values = [cand_scores[pid] for pid in shared]

    bootstrap = paired_bootstrap_ci(
        baseline_values, candidate_values,
        iterations=cfg.bootstrap_iterations, alpha=cfg.alpha, seed=f"{cfg.seed}|{baseline_protocol}|{candidate_protocol}",
    )
    mcnemar = exact_mcnemar(baseline_values, candidate_values)
    randomization = sign_flip_test(
        baseline_values, candidate_values,
        iterations=cfg.randomization_iterations, seed=f"{cfg.seed}|{baseline_protocol}|{candidate_protocol}",
    )
    decision: ComparisonDecision = decide(
        bootstrap,
        non_inferiority_margin_pp=cfg.non_inferiority_margin_pp,
        equivalence_margin_pp=cfg.equivalence_margin_pp,
        margin_frozen=cfg.margin_frozen,
    )

    comparison = PairedComparison(
        label=f"{baseline.condition}:{baseline_protocol} vs {candidate.condition}:{candidate_protocol}",
        baseline_protocol=baseline_protocol,
        candidate_protocol=candidate_protocol,
        metric=metric,
        n_paired=len(shared),
        baseline_only=sorted(set(base_scores) - set(cand_scores)),
        candidate_only=sorted(set(cand_scores) - set(base_scores)),
        bootstrap=bootstrap.as_dict(),
        mcnemar=mcnemar.as_dict(),
        randomization=randomization.as_dict(),
        decision=decision.as_dict(),
    )
    comparison.strata = _stratified(base_records, cand_records, shared, cfg)
    comparison.track_disagreement = _track_disagreement(base_records, cand_records)
    comparison.audit_sample = _audit_sample(base_records, cand_records, shared, cfg)
    return comparison


def _stratified(
    base_records: list[dict[str, Any]],
    cand_records: list[dict[str, Any]],
    shared: list[str],
    cfg: ComparisonConfig,
) -> dict[str, Any]:
    """Per-stratum paired differences with a Holm-Bonferroni family correction."""
    out: dict[str, Any] = {}
    base_scores = _per_problem_scores(base_records, cfg.primary_metric)
    cand_scores = _per_problem_scores(cand_records, cfg.primary_metric)

    for key in cfg.stratify_by:
        labels = _stratum_values(base_records, key)
        groups: dict[str, list[str]] = defaultdict(list)
        for problem_id in shared:
            groups[labels.get(problem_id, "unknown")].append(problem_id)

        stratum_out: dict[str, Any] = {}
        p_values: dict[str, float] = {}
        for label, problem_ids in sorted(groups.items()):
            if len(problem_ids) < 2:
                stratum_out[label] = {"n": len(problem_ids), "note": "too few problems for an interval"}
                continue
            left = [base_scores[pid] for pid in problem_ids]
            right = [cand_scores[pid] for pid in problem_ids]
            boot = paired_bootstrap_ci(
                left, right,
                iterations=max(1000, cfg.bootstrap_iterations // 5),
                alpha=cfg.alpha,
                seed=f"{cfg.seed}|{key}|{label}",
            )
            rand = sign_flip_test(
                left, right,
                iterations=max(1000, cfg.randomization_iterations // 5),
                seed=f"{cfg.seed}|{key}|{label}",
            )
            stratum_out[label] = {
                "n": len(problem_ids),
                "baseline_mean": boot.baseline_mean,
                "candidate_mean": boot.candidate_mean,
                "difference_pp": round(100 * boot.observed_difference, 4),
                "ci_pp": [round(100 * boot.ci_low, 4), round(100 * boot.ci_high, 4)],
                "p_value_two_sided": rand.p_value_two_sided,
            }
            p_values[label] = rand.p_value_two_sided
        if p_values:
            corrected = holm_bonferroni(p_values, alpha=cfg.alpha)
            for label, values in corrected.items():
                stratum_out[label]["holm"] = values
        out[key] = stratum_out
    return out


def _track_disagreement(base_records: list[dict[str, Any]], cand_records: list[dict[str, Any]]) -> dict[str, Any]:
    """How much of any measured difference is formatting rather than reasoning."""
    def counts(records: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(records)
        strict = sum(1 for r in records if r.get("strict_correct"))
        lenient = sum(1 for r in records if r.get("lenient_correct"))
        lenient_only = sum(1 for r in records if r.get("lenient_correct") and not r.get("strict_correct"))
        return {
            "items": total,
            "strict_accuracy": round(strict / total, 6) if total else None,
            "lenient_accuracy": round(lenient / total, 6) if total else None,
            "lenient_only_correct": lenient_only,
            "format_recoverable_rate": round(lenient_only / total, 6) if total else None,
        }

    base, cand = counts(base_records), counts(cand_records)
    payload = {"baseline": base, "candidate": cand}
    if base["strict_accuracy"] is not None and cand["strict_accuracy"] is not None:
        payload["strict_difference_pp"] = round(100 * (cand["strict_accuracy"] - base["strict_accuracy"]), 4)
    if base["lenient_accuracy"] is not None and cand["lenient_accuracy"] is not None:
        payload["lenient_difference_pp"] = round(100 * (cand["lenient_accuracy"] - base["lenient_accuracy"]), 4)
    if "strict_difference_pp" in payload and "lenient_difference_pp" in payload:
        payload["format_attributable_pp"] = round(
            payload["strict_difference_pp"] - payload["lenient_difference_pp"], 4
        )
        payload["interpretation"] = (
            "A large |format_attributable_pp| means the strict difference is driven by contract compliance "
            "rather than reasoning; report both tracks in that case."
        )
    return payload


def _audit_sample(
    base_records: list[dict[str, Any]],
    cand_records: list[dict[str, Any]],
    shared: list[str],
    cfg: ComparisonConfig,
) -> list[dict[str, Any]]:
    """Deterministic sample of problems where the two arms disagree."""
    base_index = _record_index(base_records)
    cand_index = _record_index(cand_records)
    disagreements = []
    for problem_id in shared:
        base_correct = all(r.get(cfg.primary_metric) for r in base_index[problem_id])
        cand_correct = all(r.get(cfg.primary_metric) for r in cand_index[problem_id])
        if base_correct != cand_correct:
            disagreements.append(problem_id)
    disagreements.sort(key=lambda pid: stable_shuffle_key(pid, f"compare-audit|{cfg.seed}"))
    sample = []
    for problem_id in disagreements[: cfg.audit_sample_size]:
        base_item = base_index[problem_id][0]
        cand_item = cand_index[problem_id][0]
        sample.append(
            {
                "problem_id": problem_id,
                "category": base_item.get("category"),
                "gold": base_item.get("gold"),
                "baseline_prediction": base_item.get("prediction"),
                "baseline_failure": base_item.get("failure_type"),
                "candidate_prediction": cand_item.get("prediction"),
                "candidate_failure": cand_item.get("failure_type"),
            }
        )
    return sample


def compare_runs(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    cfg: LabConfig,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Full comparison across every configured protocol pair."""
    baseline = load_run(baseline_dir)
    candidate = load_run(candidate_dir)
    comparison_cfg = cfg.comparison

    pairs: list[tuple[str, str]] = []
    for raw in comparison_cfg.protocol_pairs:
        left, right = raw.split(":")
        pairs.append((left, right))
    if not pairs:
        shared_protocols = sorted(set(baseline.protocols()) & set(candidate.protocols()))
        pairs = [(name, name) for name in shared_protocols]
        if comparison_cfg.default_end_to_end_pair:
            left, right = comparison_cfg.default_end_to_end_pair.split(":")
            if left in baseline.protocols() and right in candidate.protocols() and (left, right) not in pairs:
                pairs.insert(0, (left, right))

    results: dict[str, Any] = {}
    for baseline_protocol, candidate_protocol in pairs:
        comparison = compare_protocol_pair(baseline, candidate, baseline_protocol, candidate_protocol, comparison_cfg)
        if comparison is not None:
            results[f"{baseline_protocol}__vs__{candidate_protocol}"] = comparison.as_dict()

    report: dict[str, Any] = {
        "lab_version": __version__,
        "generated_at": time.time(),
        "baseline": baseline.provenance(),
        "candidate": candidate.provenance(),
        "settings": {
            "primary_metric": comparison_cfg.primary_metric,
            "alpha": comparison_cfg.alpha,
            "bootstrap_iterations": comparison_cfg.bootstrap_iterations,
            "randomization_iterations": comparison_cfg.randomization_iterations,
            "non_inferiority_margin_pp": comparison_cfg.non_inferiority_margin_pp,
            "equivalence_margin_pp": comparison_cfg.equivalence_margin_pp,
            "margin_frozen": comparison_cfg.margin_frozen,
            "margin_justification": comparison_cfg.margin_justification,
            "pair_on": comparison_cfg.pair_on,
            "stratify_by": comparison_cfg.stratify_by,
            "seed": comparison_cfg.seed,
        },
        "comparisons": results,
        "validity_warnings": _validity_warnings(baseline, candidate, comparison_cfg),
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        write_json(output_dir / "comparison_report.json", report)
        (output_dir / "comparison_report.md").write_text(render_comparison_markdown(report), encoding="utf-8")
        LOGGER.info("Wrote %s", output_dir / "comparison_report.json")
    return report


def _validity_warnings(baseline: RunView, candidate: RunView, cfg: ComparisonConfig) -> list[str]:
    warnings: list[str] = []
    if baseline.is_reference or candidate.is_reference:
        warnings.append(
            "At least one run used the reference backend. This comparison exercises the statistics code; "
            "it is not evidence about the hypothesis."
        )
    if not cfg.margin_frozen:
        warnings.append(
            "comparison.margin_frozen is false. The shipped 2.0 pp margin is a placeholder; freeze and "
            "justify the real margin in docs/PREREGISTRATION.md before opening TEST results."
        )
    base_ids = baseline.report.get("eval_set", {}).get("problem_ids_sha256")
    cand_ids = candidate.report.get("eval_set", {}).get("problem_ids_sha256")
    if base_ids and cand_ids and base_ids != cand_ids:
        warnings.append(
            "The two runs evaluated different problem sets (problem_ids_sha256 differ). Pairing falls back to "
            "the intersection; re-run both arms on identical held-out problem ids for the primary result."
        )
    base_split = baseline.report.get("eval_set", {}).get("split")
    cand_split = candidate.report.get("eval_set", {}).get("split")
    if base_split != cand_split:
        warnings.append(f"Split mismatch: baseline={base_split}, candidate={cand_split}.")
    if baseline.condition == candidate.condition:
        warnings.append(f"Both runs report condition {baseline.condition!r}; check that the right runs were paired.")
    return warnings


def render_comparison_markdown(report: dict[str, Any]) -> str:
    settings = report["settings"]
    lines = [
        "# FULL vs SPLIT comparison",
        "",
        f"- baseline: `{report['baseline']['name']}` (condition `{report['baseline']['condition']}`)",
        f"- candidate: `{report['candidate']['name']}` (condition `{report['candidate']['condition']}`)",
        f"- primary metric: `{settings['primary_metric']}`, paired on `{settings['pair_on']}`",
        f"- alpha: {settings['alpha']}, bootstrap: {settings['bootstrap_iterations']}, "
        f"randomization: {settings['randomization_iterations']}",
        f"- non-inferiority margin: {settings['non_inferiority_margin_pp']} pp "
        f"({'frozen' if settings['margin_frozen'] else 'PLACEHOLDER — NOT FROZEN'})",
        "",
    ]
    for warning in report["validity_warnings"]:
        lines.append(f"> ⚠️ {warning}")
    if report["validity_warnings"]:
        lines.append("")

    lines += [
        "## Paired results",
        "",
        "| comparison | n | baseline | candidate | diff (pp) | 95% CI (pp) | McNemar p | sign-flip p | decision |",
        "|---|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for key, comparison in report["comparisons"].items():
        boot = comparison["bootstrap"]
        mcnemar = comparison["mcnemar"]
        rand = comparison["randomization"]
        decision = comparison["decision"]
        verdict = decision["superiority"]
        if verdict == "not_established":
            verdict = decision["equivalence"] if decision["equivalence"] != "not_established" else decision["non_inferiority"]
        lines.append(
            f"| `{key}` | {comparison['n_paired']} | {100 * boot['baseline_mean']:.2f}% | "
            f"{100 * boot['candidate_mean']:.2f}% | {boot['observed_difference_pp']:+.2f} | "
            f"[{boot['ci_low_pp']:+.2f}, {boot['ci_high_pp']:+.2f}] | "
            f"{_p(mcnemar.get('p_value_two_sided'))} | {_p(rand.get('p_value_two_sided'))} | {verdict} |"
        )

    for key, comparison in report["comparisons"].items():
        lines += ["", f"### {key}", ""]
        for line in comparison["decision"]["rationale"]:
            lines.append(f"- {line}")
        disagreement = comparison.get("track_disagreement", {})
        if "format_attributable_pp" in disagreement:
            lines += [
                "",
                f"- strict difference: {disagreement['strict_difference_pp']:+.2f} pp",
                f"- lenient difference: {disagreement['lenient_difference_pp']:+.2f} pp",
                f"- attributable to formatting: {disagreement['format_attributable_pp']:+.2f} pp",
            ]
        if not comparison["mcnemar"]["applicable"]:
            lines.append(f"- McNemar not applicable: {comparison['mcnemar']['note']}")
    lines.append("")
    return "\n".join(lines)


def _p(value: float | None) -> str:
    if value is None:
        return "n/a"
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"
