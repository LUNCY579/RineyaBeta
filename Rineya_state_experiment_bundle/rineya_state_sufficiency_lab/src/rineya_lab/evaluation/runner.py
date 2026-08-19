"""Evaluation orchestration: run protocols, persist records, write the report."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rineya_lab.config import LabConfig
from rineya_lab.data.loading import RineyaDataset
from rineya_lab.evaluation.aggregate import (
    audit_sample,
    free_rollout_gap,
    state_sufficiency_contrast,
    summarize_protocol,
)
from rineya_lab.evaluation.backends import build_backend
from rineya_lab.evaluation.protocols import EvalRecord, build_eval_set, run_protocol
from rineya_lab.utils.determinism import seed_everything
from rineya_lab.utils.envinfo import collect_environment
from rineya_lab.utils.io import write_json, write_jsonl
from rineya_lab.utils.logging import get_logger, setup_logging
from rineya_lab.version import __version__

LOGGER = get_logger("evaluation.runner")


def evaluate(cfg: LabConfig, dataset: RineyaDataset | None = None) -> dict[str, Any]:
    """Run every configured protocol and write a complete evaluation report."""
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=output_dir / "evaluate.log")

    seed_record = seed_everything(cfg.experiment.seed)
    dataset = dataset or RineyaDataset.from_config(cfg.dataset)
    eval_set = build_eval_set(dataset, cfg)
    backend = build_backend(cfg)

    report: dict[str, Any] = {
        "lab_version": __version__,
        "experiment": asdict(cfg.experiment),
        "resolved_config": cfg.to_dict(),
        "config_fingerprint": cfg.fingerprint(),
        "seeding": seed_record,
        "environment": collect_environment(),
        "dataset": dataset.provenance(splits=(cfg.evaluation.split,)),
        "backend": backend.describe(),
        "eval_set": {
            "split": cfg.evaluation.split,
            "problems": len(eval_set),
            "problem_ids_sha256": _problem_ids_hash(eval_set.order),
            "transitions": sum(t.length for t in eval_set.iter_trajectories()),
            "limit": cfg.evaluation.limit,
            "categories": cfg.evaluation.categories or "all",
        },
        "started_at": time.time(),
        "protocols": {},
    }
    if backend.is_reference:
        report["warning"] = (
            "This report was produced with the reference backend. It validates the evaluation "
            "harness only and is not a result about the hypothesis."
        )

    all_records: list[EvalRecord] = []
    for protocol in cfg.evaluation.protocols:
        started = time.time()
        LOGGER.info("Running protocol %s", protocol)
        records = list(run_protocol(backend, eval_set, cfg, protocol))
        summary = summarize_protocol(protocol, records)
        summary["wall_clock_seconds"] = round(time.time() - started, 3)
        report["protocols"][protocol] = summary
        all_records.extend(records)

        if cfg.evaluation.save_per_example_records:
            write_jsonl(
                output_dir / "records" / f"{protocol}.jsonl",
                (r.as_dict(include_raw=cfg.evaluation.save_raw_generations) for r in records),
            )
        LOGGER.info(
            "%s: strict=%.4f lenient=%.4f over %d items",
            protocol,
            summary["strict"]["accuracy"] or 0.0,
            summary["lenient"]["accuracy"] or 0.0,
            summary["total_items"],
        )

    report["derived"] = {
        "free_rollout_gap": free_rollout_gap(report["protocols"]),
        "state_sufficiency": state_sufficiency_contrast(report["protocols"]),
    }
    report["audit_sample"] = audit_sample(all_records, cfg.evaluation.audit_sample_size, cfg.experiment.seed)
    report["finished_at"] = time.time()
    report["total_items"] = len(all_records)

    write_json(output_dir / "eval_report.json", report)
    (output_dir / "eval_report.md").write_text(render_markdown(report), encoding="utf-8")
    backend.close()
    LOGGER.info("Wrote %s", output_dir / "eval_report.json")
    return report


def _problem_ids_hash(problem_ids: list[str]) -> str:
    from rineya_lab.utils.hashing import sha256_text

    return sha256_text("|".join(sorted(problem_ids)))


def render_markdown(report: dict[str, Any]) -> str:
    """Human-readable summary. The JSON report remains the machine-readable source."""
    experiment = report["experiment"]
    lines = [
        f"# Evaluation report — {experiment['name']}",
        "",
        f"- condition: `{experiment['condition']}`",
        f"- seed: `{experiment['seed']}`",
        f"- backend: `{report['backend'].get('backend')}`"
        + (f" (mode `{report['backend'].get('reference_mode')}`)" if report["backend"].get("is_reference") else ""),
        f"- split: `{report['eval_set']['split']}`",
        f"- problems: {report['eval_set']['problems']}",
        f"- transitions: {report['eval_set']['transitions']}",
        f"- config fingerprint: `{report['config_fingerprint'][:16]}`",
        "",
    ]
    if report.get("warning"):
        lines += [f"> **{report['warning']}**", ""]

    lines += [
        "## Protocol results",
        "",
        "| protocol | n | strict | lenient | format fail | control fail | reasoning fail |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in report["protocols"].items():
        groups = summary["failure_groups"]
        lines.append(
            f"| `{name}` | {summary['total_items']} | "
            f"{_pct(summary['strict']['accuracy'])} | {_pct(summary['lenient']['accuracy'])} | "
            f"{_pct(groups['format_rate'])} | {_pct(groups['control_rate'])} | {_pct(groups['reasoning_rate'])} |"
        )

    derived = report.get("derived", {})
    if derived.get("state_sufficiency"):
        suff = derived["state_sufficiency"]
        lines += [
            "",
            "## State sufficiency",
            "",
            f"- compact state: {_pct(suff['compact_state_accuracy'])}",
            f"- complete history: {_pct(suff['full_history_accuracy'])}",
            f"- compact − history: {_pp(suff['compact_minus_history'])}",
            f"- no state (lower bound): {_pct(suff['no_state_accuracy'])}",
            f"- state only, no problem: {_pct(suff['last_state_only_accuracy'])}",
            f"- shuffled state: {_pct(suff['shuffled_state_accuracy'])}",
            f"- distractor state: {_pct(suff['distractor_state_accuracy'])}",
        ]
    if derived.get("free_rollout_gap"):
        gap = derived["free_rollout_gap"]
        lines += [
            "",
            "## Error propagation",
            "",
            f"- oracle-state transition accuracy: {_pct(gap['oracle_transition_strict_accuracy'])}",
            f"- free-rollout final accuracy: {_pct(gap['generated_rollout_strict_accuracy'])}",
            f"- free-rollout gap: {_pp(gap['gap'])}",
            f"- teacher-forced final accuracy: {_pct(gap['teacher_forced_final_accuracy'])}",
            f"- teacher-forcing vs free-rollout gap: {_pp(gap['teacher_forcing_vs_free_rollout_gap'])}",
        ]

    lines += ["", "## Failure distribution", ""]
    for name, summary in report["protocols"].items():
        lines.append(f"**{name}** — " + ", ".join(f"`{k}`={v}" for k, v in summary["failure_distribution"].items()))
    lines += [
        "",
        "## Notes",
        "",
        "- Strict is the primary track. Unparseable, refused and empty responses remain in the denominator.",
        "- Lenient is diagnostic only: it measures recoverable knowledge under formatting noise.",
        f"- {len(report.get('audit_sample', []))} records were sampled for manual audit (see `eval_report.json`).",
        "",
    ]
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def _pp(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:+.2f} pp"
