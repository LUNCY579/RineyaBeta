"""Build the arm x seed table and every paired contrast from whatever runs exist.

One command, no arguments needed, safe to run at any time: it reports on completed
runs and says plainly which are still missing. Written so a result is readable
without reconstructing the analysis by hand, and so a batch that dies part-way still
yields whatever it earned.

    python scripts/summarize_runs.py                     # every g3_* run
    python scripts/summarize_runs.py --prefix g2_budget_  # a different batch
    python scripts/summarize_runs.py --json out.json      # machine-readable too

Contrasts are paired by problem id and quote the CLUSTERED interval (unit =
generator) when the prediction files carry it, because problems sharing a template
are correlated and the i.i.d. interval is anti-conservative.

CLAIM BOUNDARY, printed with every report: none of this is confirmatory. Read the
per-contrast seed count. A single seed at 800 DEV carries ~57% power at the measured
6.0% discordance, so a null from one seed is not evidence of absence.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from dast.compare import compare_prediction_files

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def _identify(directory: Path) -> tuple[str, str] | None:
    """(arm, seed) from the run manifest, which is authoritative.

    Deliberately not parsed from the directory name: a name only encodes what the
    launcher intended, while the manifest records what the run actually did. A
    mislabelled directory would otherwise be silently grouped under the wrong arm.
    """

    try:
        manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
        return str(manifest["arm"]), str(manifest["seed"])
    except Exception:  # noqa: BLE001
        return None


def _complete(directory: Path) -> bool:
    return (directory / "dev_summary.json").exists() and (
        directory / "generation_summary.json"
    ).exists()


def _load(directory: Path) -> dict:
    dev = json.loads((directory / "dev_summary.json").read_text(encoding="utf-8"))
    gen = json.loads((directory / "generation_summary.json").read_text(encoding="utf-8"))
    return {
        "accuracy": gen["accuracy"],
        "records": gen["records"],
        "order_margin": dev.get("state/order_margin"),
        "top1": dev.get("state/checkpoint_top1"),
        "monotonic": dev.get("state/decoded_progress_monotonic"),
        "l_state": dev.get("loss/state"),
        "ppl": dev.get("lm/perplexity"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise a DAST run batch")
    parser.add_argument("--prefix", default="g3_")
    parser.add_argument("--baseline-arm", default="cot_baseline")
    parser.add_argument("--json", dest="json_out")
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args(argv)

    found: dict[tuple[str, str], dict] = {}
    pending: list[str] = []
    for directory in sorted(RUNS.glob(f"{args.prefix}*")):
        if not directory.is_dir():
            continue
        if not _complete(directory):
            pending.append(directory.name)
            continue
        identity = _identify(directory)
        if identity is None:
            print(f"  !! {directory.name}: unreadable run_manifest.json, skipped")
            continue
        arm, seed = identity
        found[(arm, seed)] = {**_load(directory), "dir": directory}

    if not found:
        print(f"no completed runs matching {args.prefix}*")
        if pending:
            print("in progress / incomplete: " + ", ".join(pending))
        return 0

    arms = sorted({arm for arm, _ in found})
    seeds = sorted({seed for _, seed in found}, key=int)

    print(f"=== {args.prefix}* : {len(found)} completed run(s), "
          f"{len(pending)} incomplete ===")
    if pending:
        print("    incomplete: " + ", ".join(pending))
    print()

    # ---- per-arm table, and the mean over whatever seeds exist ----
    header = f"{'arm':24s}" + "".join(f"{'s'+s:>10}" for s in seeds) + f"{'mean':>10}{'n':>4}"
    print(header)
    print("-" * len(header))
    per_arm_mean: dict[str, float] = {}
    for arm in arms:
        cells, values = "", []
        for seed in seeds:
            entry = found.get((arm, seed))
            if entry:
                values.append(entry["accuracy"])
                cells += f"{entry['accuracy']:>10.4f}"
            else:
                cells += f"{'-':>10}"
        mean = sum(values) / len(values)
        per_arm_mean[arm] = mean
        print(f"{arm:24s}{cells}{mean:>10.4f}{len(values):>4d}")
    print()

    # ---- mechanism metrics, seed-averaged ----
    print(f"{'arm':24s}{'margin':>10}{'top1':>8}{'monot':>8}{'L_state':>9}{'ppl':>9}")
    print("-" * 68)
    def avg(entries: list[dict], key: str) -> str:
        vals = [e[key] for e in entries if e.get(key) is not None]
        return f"{sum(vals)/len(vals):.4f}" if vals else "-"

    for arm in arms:
        rows = [found[(arm, s)] for s in seeds if (arm, s) in found]
        print(f"{arm:24s}{avg(rows,'order_margin'):>10}{avg(rows,'top1'):>8}"
              f"{avg(rows,'monotonic'):>8}{avg(rows,'l_state'):>9}{avg(rows,'ppl'):>9}")
    print()

    # ---- paired contrasts, per seed and pooled across seeds ----
    contrasts: dict[str, list[dict]] = defaultdict(list)
    for arm in arms:
        if arm == args.baseline_arm:
            continue
        for seed in seeds:
            base_entry = found.get((args.baseline_arm, seed))
            cand_entry = found.get((arm, seed))
            if not (base_entry and cand_entry):
                continue
            base, cand = base_entry["dir"], cand_entry["dir"]
            try:
                report = compare_prediction_files(
                    base / "generation_predictions.jsonl",
                    cand / "generation_predictions.jsonl",
                    bootstrap_samples=args.bootstrap,
                    seed=17,
                )
            except Exception as exc:  # noqa: BLE001 - a bad pair must not kill the report
                print(f"  !! {arm} vs {args.baseline_arm} s{seed}: {exc}")
                continue
            # NB: report carries its own "seed" (the bootstrap seed), so the run
            # seed must use a distinct key or it is silently overwritten.
            contrasts[arm].append({"run_seed": seed, **report})

    if contrasts:
        print(f"=== paired contrasts against {args.baseline_arm} ===")
        print(f"{'contrast':34s}{'diff':>9}{'clustered 95% CI':>24}{'disc':>9}{'p':>8}")
        print("-" * 84)
        for arm, rows in contrasts.items():
            for row in rows:
                cl = row["bootstrap_95_ci_clustered"]
                ci = (f"[{cl['bootstrap_95_ci'][0]:+.4f},{cl['bootstrap_95_ci'][1]:+.4f}]"
                      if cl["available"] else "iid only")
                disc = (f"{row['discordant']['candidate_only_correct']}:"
                        f"{row['discordant']['baseline_only_correct']}")
                label = f"{arm} - {args.baseline_arm} s{row['run_seed']}"
                print(f"{label:34s}"
                      f"{row['candidate_minus_baseline']:>+9.4f}{ci:>24}{disc:>9}"
                      f"{row['mcnemar_exact_two_sided_p']:>8.4f}")
            # Pooled: sum discordant counts across seeds, one exact test.
            if len(rows) > 1:
                from dast.compare import _exact_two_sided_binomial

                cand_only = sum(r["discordant"]["candidate_only_correct"] for r in rows)
                base_only = sum(r["discordant"]["baseline_only_correct"] for r in rows)
                mean_diff = sum(r["candidate_minus_baseline"] for r in rows) / len(rows)
                pooled_p = _exact_two_sided_binomial(
                    min(cand_only, base_only), cand_only + base_only
                )
                pooled_label = f"{arm} - {args.baseline_arm}  POOLED({len(rows)} seeds)"
                print(f"{pooled_label:34s}"
                      f"{mean_diff:>+9.4f}{'':>24}{str(cand_only)+':'+str(base_only):>9}"
                      f"{pooled_p:>8.4f}")
        print()

    print("CLAIM BOUNDARY: not confirmatory. Check the seed count on every contrast --")
    print("one seed at 800 DEV carries ~57% power at the measured 6.0% discordance, so a")
    print("null from a single seed is not evidence of absence. TEST is sealed.")

    if args.json_out:
        payload = {
            "prefix": args.prefix,
            "runs": {f"{a}_s{s}": {k: v for k, v in e.items() if k != "dir"}
                     for (a, s), e in found.items()},
            "incomplete": pending,
            "arm_mean_accuracy": per_arm_mean,
            "contrasts": {k: v for k, v in contrasts.items()},
        }
        Path(args.json_out).write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
