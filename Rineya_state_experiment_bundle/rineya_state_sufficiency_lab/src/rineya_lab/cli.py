"""Command line interface.

    rineya-lab validate-data --dataset-root ...
    rineya-lab plan       -c configs/train_split_stateful.yaml
    rineya-lab budget     -c configs/train_full.yaml -c2 configs/train_split_stateful.yaml
    rineya-lab train      -c configs/train_full.yaml
    rineya-lab evaluate   -c configs/eval_split_stateful.yaml
    rineya-lab compare    --baseline runs/eval_full --candidate runs/eval_split
    rineya-lab smoke      --all

Every subcommand accepts ``--set section.key=value`` overrides and writes a
fully resolved config into its output directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rineya_lab.config import ConfigError, LabConfig, load_config
from rineya_lab.utils.io import write_json
from rineya_lab.utils.logging import get_logger, setup_logging
from rineya_lab.version import __version__

LOGGER = get_logger("cli")


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-c", "--config", type=Path, help="YAML config (supports 'extends:')")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="Dotted config override, repeatable")
    parser.add_argument("--dataset-root", type=Path, help="Shortcut for --set dataset.root=...")
    parser.add_argument("--output-dir", type=Path, help="Shortcut for --set experiment.output_dir=...")
    parser.add_argument("--seed", type=int, help="Shortcut for --set experiment.seed=...")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--print-config", action="store_true", help="Print the resolved config and exit")


def _build_config(args: argparse.Namespace) -> LabConfig:
    overrides = list(args.overrides)
    if getattr(args, "dataset_root", None):
        overrides.append(f"dataset.root={args.dataset_root}")
    if getattr(args, "output_dir", None):
        overrides.append(f"experiment.output_dir={args.output_dir}")
    if getattr(args, "seed", None) is not None:
        overrides.append(f"experiment.seed={args.seed}")
    cfg = load_config(args.config, overrides)
    if args.print_config:
        print(json.dumps(cfg.to_dict(), indent=2, sort_keys=True))
        raise SystemExit(0)
    return cfg


def _dataset(cfg: LabConfig):
    from rineya_lab.data.loading import RineyaDataset

    return RineyaDataset.from_config(cfg.dataset)


def _persist_resolved(cfg: LabConfig) -> None:
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "resolved_config.json", {"config": cfg.to_dict(), "fingerprint": cfg.fingerprint(),
                                                    "argv": sys.argv, "lab_version": __version__})


# --------------------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------------------


def cmd_validate_data(args: argparse.Namespace) -> int:
    from rineya_lab.data.validate import validate_dataset

    cfg = _build_config(args)
    dataset = _dataset(cfg)
    splits = tuple(args.splits) if args.splits else ("train", "dev", "test")
    report = validate_dataset(dataset, splits=splits, schema_check=not args.no_schema, limit=args.limit)
    payload = report.as_dict()
    payload["dataset"] = dataset.provenance(splits=splits, include_hashes=not args.skip_hashes)
    if args.output:
        write_json(args.output, payload)
    print(json.dumps({k: payload[k] for k in ("status", "counts", "checks", "error_count", "warning_count")},
                     indent=2, sort_keys=True))
    for error in payload["errors"][:20]:
        print(f"  ERROR: {error}", file=sys.stderr)
    return 0 if report.status == "PASS" else 1


def cmd_plan(args: argparse.Namespace) -> int:
    from rineya_lab.data.views import build_view_plan, order_diagnostics

    cfg = _build_config(args)
    dataset = _dataset(cfg)
    plan = build_view_plan(dataset, cfg)
    payload = {"plan": plan.summary(), "order_diagnostics": order_diagnostics(plan)}
    if args.output:
        write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_budget(args: argparse.Namespace) -> int:
    from rineya_lab.data.budget import matched_budget_recommendation, measure_view
    from rineya_lab.training.loop import _load_tokenizer

    cfg = _build_config(args)
    dataset = _dataset(cfg)
    tokenizer = _load_tokenizer(cfg)
    first = measure_view(dataset, cfg, tokenizer, sample_limit=args.sample)
    payload: dict[str, Any] = {"primary": first}

    if args.config2:
        other = load_config(args.config2, args.overrides)
        second = measure_view(dataset, other, tokenizer, sample_limit=args.sample)
        payload["secondary"] = second
        payload["matched_budget"] = matched_budget_recommendation(first, second)
    if args.output:
        write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from rineya_lab.training.loop import format_manifest_summary, train

    cfg = _build_config(args)
    _persist_resolved(cfg)
    manifest = train(cfg, _dataset(cfg))
    print(format_manifest_summary(manifest))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from rineya_lab.evaluation.runner import evaluate

    cfg = _build_config(args)
    _persist_resolved(cfg)
    report = evaluate(cfg, _dataset(cfg))
    for protocol, summary in report["protocols"].items():
        strict = summary["strict"]["accuracy"]
        lenient = summary["lenient"]["accuracy"]
        print(f"{protocol:26s} strict={_fmt(strict)} lenient={_fmt(lenient)} n={summary['total_items']}")
    if report.get("derived", {}).get("free_rollout_gap"):
        gap = report["derived"]["free_rollout_gap"]["gap"]
        print(f"{'free_rollout_gap':26s} {_fmt(gap)}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from rineya_lab.analysis.compare import compare_runs

    cfg = _build_config(args)
    baseline = args.baseline or cfg.comparison.baseline_run
    candidate = args.candidate or cfg.comparison.candidate_run
    if not baseline or not candidate:
        raise SystemExit("compare needs --baseline and --candidate (or comparison.baseline_run/candidate_run)")
    output_dir = args.output_dir or Path(cfg.experiment.output_dir)
    report = compare_runs(baseline, candidate, cfg, output_dir=output_dir)
    from rineya_lab.analysis.compare import render_comparison_markdown

    print(render_comparison_markdown(report))
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    from rineya_lab.analysis.multiseed import aggregate_reports, render_markdown

    cfg = _build_config(args)
    output_dir = args.output_dir or Path(cfg.experiment.output_dir)
    payload = aggregate_reports(args.reports, cfg.comparison, output_dir=output_dir)
    print(render_markdown(payload))
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    from rineya_lab.smoke import run_smoke

    cfg = _build_config(args) if args.config else None
    results = run_smoke(
        cfg,
        dataset_root=str(args.dataset_root) if args.dataset_root else None,
        include_training=args.all or args.training,
        output_dir=args.output_dir,
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if results["status"] == "PASS" else 1


def cmd_info(args: argparse.Namespace) -> int:
    from rineya_lab.utils.envinfo import collect_environment

    payload: dict[str, Any] = {"lab_version": __version__, "environment": collect_environment()}
    try:
        cfg = _build_config(args)
        dataset = _dataset(cfg)
        payload["dataset"] = dataset.provenance(include_hashes=False)
    except (ConfigError, FileNotFoundError, RuntimeError) as exc:
        payload["dataset_error"] = str(exc)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:6.2f}%"


# --------------------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rineya-lab", description="Rineya State Sufficiency Lab")
    parser.add_argument("--version", action="version", version=f"rineya-lab {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-data", help="Strictly validate the bundled dataset")
    _common(validate)
    validate.add_argument("--splits", nargs="*", choices=["train", "dev", "test"])
    validate.add_argument("--limit", type=int, help="Validate only the first N rows per file (fast check)")
    validate.add_argument("--no-schema", action="store_true", help="Skip JSON-schema validation")
    validate.add_argument("--skip-hashes", action="store_true", help="Do not hash the data files")
    validate.add_argument("-o", "--output", type=Path)
    validate.set_defaults(func=cmd_validate_data)

    plan = subparsers.add_parser("plan", help="Show the training view plan and its order diagnostics")
    _common(plan)
    plan.add_argument("-o", "--output", type=Path)
    plan.set_defaults(func=cmd_plan)

    budget = subparsers.add_parser("budget", help="Measure token/exposure budgets and propose matched limits")
    _common(budget)
    budget.add_argument("--config2", type=Path, help="Second config to match against (e.g. the SPLIT arm)")
    budget.add_argument("--sample", type=int, default=2000, help="Rows to tokenize before projecting (0 = all)")
    budget.add_argument("-o", "--output", type=Path)
    budget.set_defaults(func=cmd_budget)

    train = subparsers.add_parser("train", help="Train one adapter under one condition")
    _common(train)
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="Run the configured evaluation protocols")
    _common(evaluate)
    evaluate.set_defaults(func=cmd_evaluate)

    compare = subparsers.add_parser("compare", help="Paired statistical comparison of two evaluation runs")
    _common(compare)
    compare.add_argument("--baseline", type=Path)
    compare.add_argument("--candidate", type=Path)
    compare.set_defaults(func=cmd_compare)

    aggregate = subparsers.add_parser("aggregate", help="Combine per-seed comparison reports into one verdict")
    _common(aggregate)
    aggregate.add_argument("reports", nargs="+", type=Path,
                           help="Comparison report JSON files, or run directories containing them")
    aggregate.set_defaults(func=cmd_aggregate)

    smoke = subparsers.add_parser("smoke", help="Offline end-to-end smoke test with the reference backend")
    _common(smoke)
    smoke.add_argument("--all", action="store_true", help="Include tiny-model adapter training (requires torch)")
    smoke.add_argument("--training", action="store_true", help="Only add the training smoke step")
    smoke.set_defaults(func=cmd_smoke)

    info = subparsers.add_parser("info", help="Print environment and dataset provenance")
    _common(info)
    info.set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"Not found: {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:  # pragma: no cover
        print("Interrupted", file=sys.stderr)
        return 130


def _dispatch(command: str, argv: list[str] | None) -> int:
    return main([command, *(argv if argv is not None else sys.argv[1:])])


def main_train(argv: list[str] | None = None) -> int:
    return _dispatch("train", argv)


def main_eval(argv: list[str] | None = None) -> int:
    return _dispatch("evaluate", argv)


def main_compare(argv: list[str] | None = None) -> int:
    return _dispatch("compare", argv)


def main_validate_data(argv: list[str] | None = None) -> int:
    return _dispatch("validate-data", argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
