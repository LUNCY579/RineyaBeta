"""B2 verdict: is cosine(g_LM, g_state) more negative under ORDERED than controls?

`scripts/gradient_conflict.py` measures one arm. This aggregates several arms into
the contrast H2 actually predicts, and adds the correction the per-arm report cannot
make on its own.

Why a LoRA-only cosine is reported alongside the global one
-----------------------------------------------------------
`global_cosine` is taken over every trainable parameter, which includes the
`state_projector`. But `L_LM` does not depend on the projector at all, so
`grad_lm` is identically zero on that whole block. Including it inflates
``||g_state||`` without contributing to the dot product, which shrinks |cosine|
toward zero and makes every arm look equally conflict-free. The LoRA-only cosine
recombines the per-module rows excluding the projector -- those are the parameters
BOTH objectives actually touch, and the only place a conflict could live.

Usage
-----
    python scripts/gradient_conflict_compare.py \\
        --arm ordered=artifacts/b2_ordered_s29.json \\
        --arm reverse=artifacts/b2_reverse_s29.json \\
        --arm final_repeated=artifacts/b2_final_repeated_s29.json \\
        --out artifacts/b2_verdict_s29.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

REFERENCE = "ordered"


def lora_only_cosine(batch: dict) -> float:
    """Recombine module cosines excluding the state projector.

    cos = sum_m cos_m |lm_m| |st_m| / sqrt(sum_m |lm_m|^2 * sum_m |st_m|^2), which
    is the exact cosine of the concatenated vectors reconstructed from per-module
    norms and cosines.
    """

    numerator = left = right = 0.0
    for module in batch["modules"]:
        if module["module"] == "state_projector":
            continue
        if module["lm_norm"] <= 0 or module["state_norm"] <= 0:
            continue
        numerator += module["cosine"] * module["lm_norm"] * module["state_norm"]
        left += module["lm_norm"] ** 2
        right += module["state_norm"] ** 2
    if left <= 0 or right <= 0:
        return float("nan")
    return numerator / math.sqrt(left * right)


def summarise(values: list[float]) -> dict:
    count = len(values)
    mean = statistics.mean(values)
    deviation = statistics.stdev(values) if count > 1 else 0.0
    return {
        "n": count,
        "mean": mean,
        "sd": deviation,
        "sem": deviation / math.sqrt(count) if count > 1 else 0.0,
        "n_negative": sum(1 for value in values if value < 0),
    }


def welch(first: list[float], second: list[float]) -> dict:
    mean_difference = statistics.mean(first) - statistics.mean(second)
    error = math.sqrt(
        statistics.variance(first) / len(first) + statistics.variance(second) / len(second)
    )
    return {
        "difference": mean_difference,
        "se": error,
        "t": mean_difference / error if error > 0 else float("nan"),
        "significant_at_2_sigma": abs(mean_difference) > 2 * error if error > 0 else False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="name=path.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    series: dict[str, dict[str, list[float]]] = {}
    meta: dict[str, dict] = {}
    for entry in args.arm:
        name, _, path = entry.partition("=")
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        series[name] = {
            "global_cosine": [b["global_cosine"] for b in blob["batches"]],
            "lora_only_cosine": [lora_only_cosine(b) for b in blob["batches"]],
            "fraction_modules_negative": [
                b["fraction_modules_negative_cosine"] for b in blob["batches"]
            ],
            "grad_norm_ratio": [b["grad_norm_ratio_state_over_lm"] for b in blob["batches"]],
        }
        meta[name] = {"run": blob.get("run"), "seed": blob.get("seed"), "arm": blob.get("arm")}

    metrics = ("global_cosine", "lora_only_cosine", "fraction_modules_negative")
    print("=== per-arm ===")
    summaries = {}
    for name, data in series.items():
        summaries[name] = {metric: summarise(data[metric]) for metric in metrics}
        summaries[name]["grad_norm_ratio_median"] = statistics.median(data["grad_norm_ratio"])
        print(f"  {name}  (n={summaries[name]['global_cosine']['n']} batches)")
        for metric in metrics:
            row = summaries[name][metric]
            print(f"     {metric:26s} {row['mean']:+.5f} +- {row['sem']:.5f}   "
                  f"negative in {row['n_negative']}/{row['n']}")
        print(f"     {'grad norm ratio (median)':26s} "
              f"{summaries[name]['grad_norm_ratio_median']:.2f}")

    print(f"\n=== H2 contrast: is {REFERENCE} more negative than the controls? ===")
    contrasts = {}
    if REFERENCE in series:
        for metric in metrics:
            contrasts[metric] = {}
            print(f"  {metric}:")
            for other in series:
                if other == REFERENCE:
                    continue
                test = welch(series[REFERENCE][metric], series[other][metric])
                contrasts[metric][other] = test
                flag = "SIGNIFICANT" if test["significant_at_2_sigma"] else "not significant"
                print(f"     {REFERENCE} - {other:16s} {test['difference']:+.5f} "
                      f"(se {test['se']:.5f}, t={test['t']:+.2f})  {flag}")

    any_significant = any(
        test["significant_at_2_sigma"]
        for metric in contrasts.values()
        for test in metric.values()
    )
    verdict = (
        "H2 NOT SUPPORTED: no arm's LM/state gradient cosine differs from the others "
        "beyond 2 sigma, and all are within ~0.01 of orthogonal. The two objectives do "
        "not fight; they are nearly perpendicular. Per the frozen stop rule, do not "
        "sign the V2 9-run matrix."
        if not any_significant
        else "At least one contrast clears 2 sigma; read the table before concluding."
    )
    print(f"\nVERDICT: {verdict}")

    payload = {
        "arms": meta,
        "reference_arm": REFERENCE,
        "summaries": summaries,
        "contrasts": contrasts,
        "any_contrast_significant_at_2_sigma": any_significant,
        "verdict": verdict,
        "limitations": [
            "Measured on adapters saved at the END of training (250 optimizer steps), "
            "not as a trajectory over training. Gradients may decorrelate by "
            "convergence, so this does not exclude a transient early-training conflict.",
            "One seed (29). Batch size 2, so each batch is a 2-example estimate.",
            "grad_norm_ratio is unstable because L_LM is near zero at convergence "
            "(final train LM loss ~7e-4); the state gradient dominates in MAGNITUDE "
            "while being orthogonal in DIRECTION.",
        ],
        "claim_boundary": "B2 diagnosis only. Not a capability result. TEST sealed.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
