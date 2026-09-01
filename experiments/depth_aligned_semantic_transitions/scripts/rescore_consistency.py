"""Were all 12 V1 runs scored by the SAME scorer? Re-score them and find out.

`FINDINGS_G3_SEEDS.md` states the scorer was "frozen at f160f38". It was not.
`compare.py` reads the `correct` field baked into each `generation_predictions.jsonl`
at generation time, so a run scored before a scorer fix keeps its old verdict
forever. Three of the twelve V1 runs -- every seed-17 arm -- were generated at
commits that predate `f160f38`, the commit that made `state`-type answers
order-insensitive.

This script re-applies the CURRENT scorer to every saved prediction file and
reports what changes. It never regenerates text: the model output is fixed and
committed, only the verdict is recomputed. That makes the correction exact rather
than an estimate, and cheap rather than a re-run.

Usage
-----
    python scripts/rescore_consistency.py
    python scripts/rescore_consistency.py --json artifacts/rescore_consistency.json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dast.evaluation import answers_equal  # noqa: E402

ARMS = ("cot_baseline", "state_final_repeated", "state_ordered", "state_reverse")
SEEDS = (17, 29, 43)
SCORER_FIX = "f160f38"
CONTRASTS = (
    ("state_ordered", "state_final_repeated"),
    ("state_ordered", "cot_baseline"),
    ("state_ordered", "state_reverse"),
    ("state_reverse", "state_final_repeated"),
    ("state_reverse", "cot_baseline"),
    ("state_final_repeated", "cot_baseline"),
)


def exact_two_sided_binomial(successes: int, trials: int) -> float:
    """Exact two-sided binomial test against p=0.5, in log space so it never overflows."""

    if trials == 0:
        return 1.0
    low = min(successes, trials - successes)
    tail = sum(
        math.exp(
            math.lgamma(trials + 1)
            - math.lgamma(index + 1)
            - math.lgamma(trials - index + 1)
            - trials * math.log(2.0)
        )
        for index in range(low + 1)
    )
    return min(1.0, 2.0 * tail)


def _is_ancestor(candidate: str, descendant: str) -> bool | None:
    try:
        done = subprocess.run(
            ["git", "merge-base", "--is-ancestor", candidate, descendant],
            capture_output=True, timeout=30,
        )
        return done.returncode == 0
    except Exception:  # noqa: BLE001
        return None


def load_run(directory: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    with (directory / "generation_predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            rows[record["problem_id"]] = {
                "as_scored": bool(record["correct"]),
                "rescored": bool(
                    answers_equal(record["expected"], record["extracted"], record["answer_type"])
                ),
                "answer_type": record["answer_type"],
                "expected": record["expected"],
                "extracted": record["extracted"],
            }
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--json", default=None)
    parser.add_argument("--eval-records", type=int, default=800)
    args = parser.parse_args()

    index: dict[tuple[str, int], Path] = {}
    revisions: dict[tuple[str, int], str] = {}
    for manifest_path in sorted(Path(args.runs).glob("*/run_manifest.json")):
        directory = manifest_path.parent
        if not (directory / "generation_summary.json").exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config", {}).get("data", {}).get("max_eval_records") != args.eval_records:
            continue
        key = (manifest["arm"], manifest["seed"])
        index[key] = directory
        revisions[key] = manifest.get("environment", {}).get("git_revision", "")

    missing = [(arm, seed) for arm in ARMS for seed in SEEDS if (arm, seed) not in index]
    if missing:
        raise SystemExit(f"incomplete V1 matrix; missing {missing}")

    data = {key: load_run(path) for key, path in index.items()}

    print(f"=== provenance: was each run generated after the {SCORER_FIX} scorer fix? ===")
    provenance = {}
    for key in sorted(index, key=lambda k: (k[1], k[0])):
        revision = revisions[key]
        after = _is_ancestor(SCORER_FIX, revision)
        label = {True: "after", False: "BEFORE", None: "unknown"}[after]
        provenance[f"{key[0]}@{key[1]}"] = {
            "directory": index[key].name, "git_revision": revision, "after_scorer_fix": after,
        }
        print(f"  {key[0]:22s} seed {key[1]:>2}  {revision[:8]}  {label}")

    print(f"\n=== re-scoring {len(index)} runs with the current scorer ===")
    flips: dict[str, list[dict[str, object]]] = {}
    for key, rows in sorted(data.items(), key=lambda item: (item[0][1], item[0][0])):
        changed = [
            {"problem_id": pid, **{k: v for k, v in row.items() if k != "as_scored"},
             "as_scored": row["as_scored"]}
            for pid, row in rows.items()
            if row["as_scored"] != row["rescored"]
        ]
        if changed:
            flips[f"{key[0]}@{key[1]}"] = changed
        print(f"  {index[key].name:34s} n={len(rows):4d} flipped={len(changed):3d}")

    print("\n=== accuracy, as published -> consistently scored ===")
    accuracy = {}
    for arm in ARMS:
        cells = []
        for seed in SEEDS:
            rows = data[(arm, seed)]
            cells.append((
                sum(r["as_scored"] for r in rows.values()) / len(rows),
                sum(r["rescored"] for r in rows.values()) / len(rows),
            ))
        published = sum(a for a, _ in cells) / len(cells)
        corrected = sum(b for _, b in cells) / len(cells)
        accuracy[arm] = {
            "per_seed_published": [a for a, _ in cells],
            "per_seed_rescored": [b for _, b in cells],
            "mean_published": published,
            "mean_rescored": corrected,
        }
        marker = "  <== CHANGED" if abs(published - corrected) > 1e-9 else ""
        print(f"  {arm:22s} " + " ".join(f"{a:.4f}->{b:.4f}" for a, b in cells)
              + f"   mean {published:.4f}->{corrected:.4f}{marker}")

    print("\n=== pooled contrasts (2,400 paired problems, exact two-sided McNemar) ===")
    print(f"  {'contrast':42s} {'as published':>24s} {'consistently scored':>24s}")
    contrasts = {}
    for first, second in CONTRASTS:
        cell = {}
        for field in ("as_scored", "rescored"):
            wins = losses = 0
            delta = 0
            for seed in SEEDS:
                left, right = data[(first, seed)], data[(second, seed)]
                for pid in left:
                    if left[pid][field] and not right[pid][field]:
                        wins += 1
                    elif right[pid][field] and not left[pid][field]:
                        losses += 1
                delta += sum(left[p][field] for p in left) - sum(right[p][field] for p in right)
            total = len(SEEDS) * args.eval_records
            cell[field] = {
                "delta": delta / total,
                "discordant_favouring_first": wins,
                "discordant_favouring_second": losses,
                "p": exact_two_sided_binomial(wins, wins + losses),
            }
        contrasts[f"{first} - {second}"] = cell
        published, corrected = cell["as_scored"], cell["rescored"]
        marker = "  <== CHANGED" if abs(published["p"] - corrected["p"]) > 1e-9 else ""
        print(
            f"  {first + ' - ' + second:42s} "
            f"{published['delta']:+.4f} {published['discordant_favouring_first']:3d}:"
            f"{published['discordant_favouring_second']:3d} p={published['p']:.4f}   "
            f"{corrected['delta']:+.4f} {corrected['discordant_favouring_first']:3d}:"
            f"{corrected['discordant_favouring_second']:3d} p={corrected['p']:.4f}{marker}"
        )

    payload = {
        "scorer_fix_commit": SCORER_FIX,
        "provenance": provenance,
        "flipped_verdicts": flips,
        "accuracy": accuracy,
        "contrasts": contrasts,
        "claim_boundary": (
            "Re-scoring changes verdicts only, never generated text. DEV only; TEST sealed. "
            "The corrected numbers are the ones to cite: they apply one scorer to all 12 runs."
        ),
    }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
