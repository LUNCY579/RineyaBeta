"""What can the generated-accuracy endpoint actually detect? Exact McNemar power.

This is the arithmetic that must run BEFORE any sweep is launched, because this
endpoint is far less sensitive than its p-values suggest. `FINDINGS_G3_SEEDS.md`
section 4a records three separate occasions on which an under-powered contrast was
over-read here; the defence against a fourth is to compute the minimum detectable
effect first and compare it to the effect actually being hunted.

Power model
-----------
The endpoint is paired binary correctness. Under exact McNemar only the discordant
pairs carry information: with `n_d` discordant pairs and a probability `p` that a
discordant pair favours the better arm, the effect is `delta = n_d (2p - 1) / N`.
Power is the mass of the exact two-sided binomial rejection region under `p`. The
discordance rate is taken from the real V1 predictions rather than assumed.

Usage
-----
    python scripts/endpoint_power.py
    python scripts/endpoint_power.py --json artifacts/endpoint_power.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DEV_RECORDS = 800
PRACTICAL_MARGIN = 0.020  # PREREGISTRATION.md section 6, fixed before the runs.


def _log_choose(total: int, chosen: int) -> float:
    return math.lgamma(total + 1) - math.lgamma(chosen + 1) - math.lgamma(total - chosen + 1)


def binomial_pmf(successes: int, trials: int, probability: float) -> float:
    if successes < 0 or successes > trials:
        return 0.0
    if probability <= 0.0:
        return 1.0 if successes == 0 else 0.0
    if probability >= 1.0:
        return 1.0 if successes == trials else 0.0
    return math.exp(
        _log_choose(trials, successes)
        + successes * math.log(probability)
        + (trials - successes) * math.log1p(-probability)
    )


@lru_cache(maxsize=None)
def exact_two_sided_p(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    low = min(successes, trials - successes)
    tail = sum(
        math.exp(_log_choose(trials, index) - trials * math.log(2.0))
        for index in range(low + 1)
    )
    return min(1.0, 2.0 * tail)


@lru_cache(maxsize=None)
def _critical_boundary(discordant: int, alpha: float) -> int | None:
    """Largest b <= n/2 whose exact two-sided p is at or below alpha."""

    boundary = None
    for candidate in range(discordant // 2 + 1):
        if exact_two_sided_p(candidate, discordant) <= alpha:
            boundary = candidate
        else:
            break
    return boundary


def power(discordant: float, probability: float, alpha: float = 0.05) -> float:
    trials = int(round(discordant))
    if trials < 1:
        return 0.0
    boundary = _critical_boundary(trials, alpha)
    if boundary is None:
        return 0.0
    lower = sum(binomial_pmf(b, trials, probability) for b in range(boundary + 1))
    upper = sum(binomial_pmf(b, trials, probability) for b in range(trials - boundary, trials + 1))
    return lower + upper


def minimum_detectable_effect(
    records: int, discordance: float, alpha: float = 0.05, target: float = 0.80
) -> tuple[float, float, float]:
    discordant = records * discordance
    low, high = 0.5, 1.0
    for _ in range(50):
        middle = (low + high) / 2
        if power(discordant, middle, alpha) >= target:
            high = middle
        else:
            low = middle
    return discordant * (2 * high - 1) / records, high, discordant


def required_records(
    effect: float, discordance: float, alpha: float = 0.05, target: float = 0.80
) -> int:
    low, high = 100, 400_000
    while low < high:
        middle = (low + high) // 2
        detectable, _, _ = minimum_detectable_effect(middle, discordance, alpha, target)
        if detectable <= effect:
            high = middle
        else:
            low = middle + 1
    return high


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=None)
    parser.add_argument(
        "--discordance",
        type=float,
        default=133 / 2400,
        help="measured ordered-vs-baseline discordance rate over the pooled V1 matrix",
    )
    args = parser.parse_args()
    rate = args.discordance

    print(f"measured discordance rate = {rate:.4%} (V1 ordered vs baseline, pooled 3 seeds)")
    print(f"pre-declared practical margin = {PRACTICAL_MARGIN:.1%}\n")

    print("=== minimum detectable effect at 80% power, alpha=0.05 two-sided ===")
    mde_rows = []
    for seeds in (1, 2, 3, 5, 8, 10, 15, 20):
        records = DEV_RECORDS * seeds
        detectable, _, discordant = minimum_detectable_effect(records, rate)
        mde_rows.append({"seeds": seeds, "records": records, "mde": detectable})
        print(f"  seeds={seeds:2d}  N={records:6d}  n_discordant~{discordant:7.1f}   "
              f"MDE = {detectable * 100:5.2f} pp")

    print("\n=== records required to detect a given effect at 80% power ===")
    requirement_rows = []
    for effect, label in (
        (0.0117, "V1 ordered - final_repeated (consistently scored)"),
        (0.0100, "V1 ordered - baseline (consistently scored)"),
        (0.0075, "three quarters of the ordered deficit"),
        (0.0050, "half of the ordered deficit"),
        (0.0025, "one quarter -- a per-checkpoint share if B1 splits it four ways"),
    ):
        records = required_records(effect, rate)
        requirement_rows.append({"effect": effect, "records": records, "label": label})
        print(f"  {effect * 100:5.2f} pp  N={records:6d} = {records / DEV_RECORDS:5.1f} seeds"
              f"   ({label})")

    one_seed, _, _ = minimum_detectable_effect(DEV_RECORDS, rate)
    three_seed, _, _ = minimum_detectable_effect(3 * DEV_RECORDS, rate)
    observed = 0.0100
    print("\n=== consequences ===")
    print(f"  1-seed MDE = {one_seed * 100:.2f} pp; 3-seed MDE = {three_seed * 100:.2f} pp")
    print(f"  V1's observed ordered deficit is {observed * 100:.2f} pp, i.e. BELOW its own "
          f"3-seed MDE of {three_seed * 100:.2f} pp.")
    print(f"  Post-hoc power of the V1 ordered-baseline contrast: {power(132, 78 / 132):.0%}.")
    print(f"  A ONE-SEED pilot cannot detect the full {observed * 100:.2f} pp deficit "
          f"(needs {one_seed * 100:.2f} pp).")
    print("  => B1 single-checkpoint pilots cannot yield a CAPABILITY result at one seed, and a")
    print("     per-checkpoint share of the deficit is out of reach at any affordable seed count.")
    print("     Run B1 for L_state / gradient evidence, or not at all; do not read its accuracy.")

    payload = {
        "discordance_rate": rate,
        "practical_margin": PRACTICAL_MARGIN,
        "alpha": 0.05,
        "target_power": 0.80,
        "minimum_detectable_effect": mde_rows,
        "required_records": requirement_rows,
        "observed_ordered_baseline_deficit": observed,
        "one_seed_mde": one_seed,
        "three_seed_mde": three_seed,
        "post_hoc_power_v1_ordered_baseline": power(132, 78 / 132),
        "conclusion": (
            "The generated-accuracy endpoint cannot resolve sub-1.4 pp effects at three seeds. "
            "B1 at one seed has MDE 2.27 pp and therefore cannot produce a capability result. "
            "A higher-powered paired endpoint (Stage E) is a precondition for B1/B3 being "
            "informative about capability."
        ),
    }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
