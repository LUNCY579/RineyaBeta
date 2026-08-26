#!/usr/bin/env python3
"""Run the controlled feasibility study across independent seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))

from feasibility import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first-seed", type=int, default=20260822)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--train-steps", type=int, default=40)
    ap.add_argument("--eval-tasks", type=int, default=32)
    ap.add_argument("--out", type=Path, default=EXP / "results" / "feasibility_multiseed.json")
    args = ap.parse_args()

    reports = []
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        ns = argparse.Namespace(
            seed=seed,
            layers=6,
            rows=16,
            cols=16,
            budget=24,
            gradient_noise=1.35,
            peer_noise=0.20,
            hidden_dim=48,
            train_steps=args.train_steps,
            meta_batch=6,
            meta_lr=3e-3,
            eval_tasks=args.eval_tasks,
        )
        report = run(ns)
        reports.append(report)
        print(
            f"seed {seed}: pass={report['all_checks_pass']} "
            f"peer_loss={report['metrics']['peer_conditioned']['mean_loss']:.6f} "
            f"peer_recall={report['metrics']['peer_conditioned']['mean_support_recall']:.4f}"
        )

    arms = reports[0]["metrics"].keys()
    aggregate = {}
    for arm in arms:
        losses = [r["metrics"][arm]["mean_loss"] for r in reports]
        recalls = [r["metrics"][arm]["mean_support_recall"] for r in reports]
        aggregate[arm] = {
            "mean_loss_across_seeds": statistics.fmean(losses),
            "std_loss_across_seeds": statistics.pstdev(losses),
            "mean_support_recall_across_seeds": statistics.fmean(recalls),
            "std_support_recall_across_seeds": statistics.pstdev(recalls),
        }

    check_names = reports[0]["checks"].keys()
    pass_counts = {
        name: sum(bool(r["checks"][name]) for r in reports) for name in check_names
    }
    concise_per_seed = []
    for report in reports:
        meta_curve = report["meta_curve"]
        concise_per_seed.append(
            {
                "seed": report["seed"],
                "all_checks_pass": report["all_checks_pass"],
                "checks": report["checks"],
                "meta_loss_first": meta_curve["first"],
                "meta_loss_last": meta_curve["last"],
                "metrics": report["metrics"],
            }
        )

    output = {
        "status": "synthetic_multiseed_mathematical_feasibility_only",
        "claim_boundary": (
            "Independent-seed repetition of a controlled task whose optimum is "
            "constructed from peer structural properties; not an LLM result."
        ),
        "configuration": {
            "first_seed": args.first_seed,
            "seed_count": args.seeds,
            "train_steps": args.train_steps,
            "eval_tasks_per_seed": args.eval_tasks,
        },
        "seeds": [r["seed"] for r in reports],
        "all_seed_checks_pass": all(r["all_checks_pass"] for r in reports),
        "check_pass_counts": pass_counts,
        "aggregate": aggregate,
        "per_seed": concise_per_seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    if not output["all_seed_checks_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
