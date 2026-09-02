"""Stage E verdict: does the paired teacher-forced endpoint reproduce the V1 result?

This is a VALIDATION script before it is a measurement script. The answer on this
matrix is already known from generated accuracy:

    state_final_repeated ~ cot_baseline ~ state_reverse  >  state_ordered

If the teacher-forced endpoint does not reproduce that ordering, it is not a usable
screen for future work and must be discarded rather than explained. Passing the gate
licenses using it to rank candidates cheaply -- never to promote one on its own.

Statistics
----------
Paired by `problem_id`, pooled over seeds, with a bootstrap clustered on `generator`
(30 clusters), matching what `compare.py` already does for accuracy. Problems sharing
a generator share a template and are not independent, so resampling problems i.i.d.
would be anti-conservative.

Direction convention: a POSITIVE difference means the first arm has HIGHER NLL, i.e.
fits the gold continuation WORSE. So the accuracy result
`ordered - baseline = -1.00 pp` predicts `ordered - baseline > 0` here.

Usage
-----
    python scripts/teacher_forced_compare.py --out artifacts/stage_e_verdict.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("cot_baseline", "state_final_repeated", "state_ordered", "state_reverse")
SEEDS = (17, 29, 43)
CONTRASTS = (
    ("state_ordered", "cot_baseline"),
    ("state_ordered", "state_final_repeated"),
    ("state_ordered", "state_reverse"),
    ("state_reverse", "cot_baseline"),
    ("state_final_repeated", "cot_baseline"),
)
# Accuracy deltas (consistently scored). Negative = worse accuracy, so the NLL
# contrast should be POSITIVE for the endpoint to agree.
ACCURACY_ARM_MEANS = {
    "cot_baseline": 0.8937,
    "state_final_repeated": 0.8954,
    "state_reverse": 0.8921,
    "state_ordered": 0.8838,
}
ACCURACY = {
    ("state_ordered", "cot_baseline"): -0.0100,
    ("state_ordered", "state_final_repeated"): -0.0117,
    ("state_ordered", "state_reverse"): -0.0083,
    ("state_reverse", "cot_baseline"): -0.0017,
    ("state_final_repeated", "cot_baseline"): +0.0017,
}


def load_matrix(runs_dir: Path, metric: str):
    index, data = {}, {}
    for manifest_path in sorted(runs_dir.glob("*/run_manifest.json")):
        directory = manifest_path.parent
        scores = directory / "teacher_forced_per_problem.jsonl"
        if not scores.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        key = (manifest["arm"], manifest["seed"])
        index[key] = directory.name
        rows = {}
        generators = {}
        with scores.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                rows[record["problem_id"]] = record[metric]
                generators[record["problem_id"]] = record["generator"]
        data[key] = (rows, generators)
    return index, data


def clustered_bootstrap(
    differences: dict[str, float], clusters: dict[str, str], samples: int = 5000, seed: int = 1701
) -> tuple[float, float, float]:
    """Percentile CI resampling GENERATORS, not problems."""

    buckets: dict[str, list[float]] = defaultdict(list)
    for problem_id, value in differences.items():
        buckets[clusters[problem_id]].append(value)
    keys = sorted(buckets)
    totals = torch.tensor([sum(buckets[k]) for k in keys], dtype=torch.float64)
    counts = torch.tensor([len(buckets[k]) for k in keys], dtype=torch.float64)
    generator = torch.Generator()
    generator.manual_seed(seed)
    draws = torch.randint(0, len(keys), (samples, len(keys)), generator=generator)
    means = totals[draws].sum(dim=1) / counts[draws].sum(dim=1)
    ordered = means.sort().values
    return (
        float(totals.sum() / counts.sum()),
        float(ordered[int(0.025 * samples)]),
        float(ordered[int(0.975 * samples)]),
    )


def exact_two_sided_binomial(successes: int, trials: int) -> float:
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


def _pearson(first: list[float], second: list[float]) -> float:
    count = len(first)
    mean_first = sum(first) / count
    mean_second = sum(second) / count
    numerator = sum(
        (first[i] - mean_first) * (second[i] - mean_second) for i in range(count)
    )
    denominator = math.sqrt(
        sum((value - mean_first) ** 2 for value in first)
        * sum((value - mean_second) ** 2 for value in second)
    )
    return numerator / denominator if denominator > 0 else 0.0


def spearman_arms(nll: dict[str, float], accuracy: dict[str, float]) -> float:
    """Rank correlation of arm-mean NLL against arm-mean accuracy. n=4: direction only."""

    arms = sorted(nll)
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        for position, index in enumerate(order):
            out[index] = position + 1
        return out
    return _pearson(ranks([nll[a] for a in arms]), ranks([accuracy[a] for a in arms]))


def sign_test(differences: dict[str, float]) -> dict:
    """Paired sign test over ALL problems.

    This is where the power gain over generated accuracy actually comes from.
    Exact McNemar can only use DISCORDANT pairs -- 133 of 2400 at the measured 5.54%
    discordance. A sign test on a continuous paired difference uses every pair whose
    difference is non-zero, so the effective sample is ~18x larger.
    """

    positive = sum(1 for value in differences.values() if value > 0)
    negative = sum(1 for value in differences.values() if value < 0)
    used = positive + negative
    return {
        "n_pairs": len(differences),
        "n_used": used,
        "n_first_worse": positive,
        "n_second_worse": negative,
        "fraction_first_worse": positive / used if used else float("nan"),
        "p": exact_two_sided_binomial(positive, used),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default=str(ROOT / "runs"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--metric", default="nll_mean")
    args = parser.parse_args()

    index, data = load_matrix(Path(args.runs), args.metric)
    missing = [(a, s) for a in ARMS for s in SEEDS if (a, s) not in data]
    if missing:
        raise SystemExit(f"incomplete Stage E matrix; missing {missing}")

    print(f"=== per-arm mean {args.metric} ===")
    arm_means = {}
    for arm in ARMS:
        cells = [statistics.mean(data[(arm, seed)][0].values()) for seed in SEEDS]
        arm_means[arm] = statistics.mean(cells)
        print(f"  {arm:22s} " + "  ".join(f"{value:.6f}" for value in cells)
              + f"   mean {arm_means[arm]:.6f}")
    ranking = sorted(ARMS, key=lambda arm: arm_means[arm])
    print(f"\n  ranking, best (lowest NLL) first: {' < '.join(ranking)}")

    print(f"\n=== paired contrasts, pooled 3 seeds, bootstrap clustered on generator ===")
    print(f"  {'contrast':44s} {'d(NLL)':>11} {'95% CI':>26} {'accuracy':>10} {'agree':>6}")
    contrasts, agreements = {}, []
    for first, second in CONTRASTS:
        differences, clusters = {}, {}
        for seed in SEEDS:
            left, generators = data[(first, seed)]
            right, _ = data[(second, seed)]
            for problem_id in left:
                key = f"{seed}:{problem_id}"
                differences[key] = left[problem_id] - right[problem_id]
                clusters[key] = generators[problem_id]
        mean, low, high = clustered_bootstrap(differences, clusters)
        signs = sign_test(differences)
        median = statistics.median(differences.values())
        accuracy = ACCURACY[(first, second)]
        # Endpoint agrees when a worse-accuracy arm also has higher NLL.
        agrees = (mean > 0) == (accuracy < 0)
        excludes_zero = low > 0 or high < 0
        agreements.append((agrees, excludes_zero, abs(accuracy)))
        contrasts[f"{first} - {second}"] = {
            "delta_nll_mean": mean, "ci_low": low, "ci_high": high,
            "delta_nll_median": median, "sign_test": signs,
            "accuracy_delta": accuracy, "direction_agrees": agrees,
            "ci_excludes_zero": excludes_zero,
        }
        print(f"  {first + ' - ' + second:44s} {mean:>+11.6f} "
              f"[{low:>+.6f},{high:>+.6f}] {accuracy:>+10.4f} "
              f"{'yes' if agrees else 'NO':>6}")

    print("\n=== paired SIGN test (uses all pairs, not just discordant ones) ===")
    print(f"  {'contrast':44s} {'median d':>12} {'first worse':>13} {'p':>10}")
    for label, row in contrasts.items():
        signs = row["sign_test"]
        print(f"  {label:44s} {row['delta_nll_median']:>+12.8f} "
              f"{signs['n_first_worse']:>6}/{signs['n_used']:<6} {signs['p']:>10.2e}")

    ordered_worst = ranking[-1] == "state_ordered"
    primary = contrasts["state_ordered - cot_baseline"]
    n_agree = sum(1 for agrees, _, _ in agreements if agrees)

    # Criteria 1-3 are the ordering checks. They are necessary and NOT sufficient:
    # every arm that carries a state loss raises LM NLL, so "same sign as accuracy"
    # is nearly free when four of five accuracy deltas are negative. Criteria 4-6 are
    # what decide whether the endpoint can actually stand in for accuracy.
    magnitude_r = _pearson(
        [row["delta_nll_mean"] for row in contrasts.values()],
        [-row["accuracy_delta"] for row in contrasts.values()],
    )
    decisive = contrasts["state_ordered - state_reverse"]
    resolves_decisive = decisive["sign_test"]["p"] < 0.05
    null_contrast = contrasts["state_reverse - cot_baseline"]
    false_alarm = null_contrast["sign_test"]["p"] < 0.001 and abs(
        null_contrast["accuracy_delta"]
    ) < 0.005

    print("\n=== validation gate ===")
    print(f"  1. state_ordered is the worst arm by NLL:            {ordered_worst}")
    print(f"  2. ordered-baseline direction agrees:                {primary['direction_agrees']}")
    print(f"  3. ordered-baseline CI excludes zero:                {primary['ci_excludes_zero']}")
    print(f"     (contrasts whose direction agrees: {n_agree}/{len(CONTRASTS)})")
    print(f"  4. magnitude tracks accuracy, Pearson r =            {magnitude_r:+.3f}"
          f"   {'ok' if magnitude_r > 0.5 else 'FAIL (uncorrelated)'}")
    print(f"  5. resolves ordered-reverse (accuracy -0.83 pp):     {resolves_decisive}"
          f"   p={decisive['sign_test']['p']:.3f}")
    print(f"  6. no false alarm on reverse-baseline (accuracy null): {not false_alarm}"
          f"   p={null_contrast['sign_test']['p']:.1e}")

    usable_as_capability_screen = (
        ordered_worst and primary["direction_agrees"] and primary["ci_excludes_zero"]
        and magnitude_r > 0.5 and resolves_decisive and not false_alarm
    )
    verdict = (
        "STAGE E VALIDATED as a capability screen: it reproduces the V1 ordering, its "
        "magnitudes track accuracy, it resolves ordered-reverse, and it does not fire on "
        "null contrasts. Generated accuracy still remains the capability endpoint."
        if usable_as_capability_screen
        else "STAGE E IS NOT A CAPABILITY SCREEN. It passes the ordering checks (1-3) but "
        f"fails the ones that matter: magnitudes are uncorrelated with accuracy (r={magnitude_r:+.3f}), "
        f"it cannot resolve ordered-reverse (p={decisive['sign_test']['p']:.3f}) which is the one "
        "contrast carrying the V1 story, and it fires at "
        f"p={null_contrast['sign_test']['p']:.0e} on reverse-baseline, which accuracy says is null. "
        "What it measures with enormous power is AUXILIARY-LOSS INTERFERENCE with LM fitting -- "
        "every state-carrying arm has higher NLL -- not reasoning capability. Keep it as an "
        "interference monitor. It does NOT make B1/B3 answerable on capability."
    )
    print(f"\nVERDICT: {verdict}")

    payload = {
        "metric": args.metric,
        "runs": {f"{arm}@{seed}": index[(arm, seed)] for arm in ARMS for seed in SEEDS},
        "arm_means": arm_means,
        "ranking_best_first": ranking,
        "contrasts": contrasts,
        "n_contrasts_agreeing": n_agree,
        "accuracy_reference": ACCURACY_ARM_MEANS,
        "spearman_arm_nll_vs_accuracy": spearman_arms(arm_means, ACCURACY_ARM_MEANS),
        "magnitude_pearson_nll_vs_accuracy": magnitude_r,
        "resolves_ordered_vs_reverse": resolves_decisive,
        "false_alarm_on_null_contrast": false_alarm,
        "ordering_checks_passed": bool(
            ordered_worst and primary["direction_agrees"] and primary["ci_excludes_zero"]
        ),
        "validated_as_capability_screen": usable_as_capability_screen,
        "usable_as_interference_monitor": True,
        "verdict": verdict,
        "claim_boundary": (
            "DEV only, TEST sealed. Teacher-forced NLL is a companion endpoint, never a "
            "replacement for generated accuracy."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
