"""Stage C2: does pi*(x) vary stably by problem, or is the flat landscape uniform?

C1 measured a POOLED permutation landscape and found it nearly flat. A flat
average can hide stable per-example structure: if different problem families
genuinely prefer different state-to-depth assignments, a universal permutation is
the wrong object and Soft/Adaptive DAST becomes motivated. If instead the apparent
per-example preferences are noise, a router would be fitting noise and Adaptive
DAST is forbidden.

The trap this script is built to avoid
--------------------------------------
With 24 permutations and a noisy per-problem estimate, `argmin` ALWAYS looks
heterogeneous. Every problem gets some favourite permutation, and a histogram of
those favourites looks structured even under a perfectly uniform truth. So the
question is never "do preferences differ?" but "do they REPLICATE?".

Design: split-half reliability. Each replicate partitions problems into
FIT (50%) / EVAL_A (25%) / EVAL_B (25%). One shared readout is fitted per
permutation on FIT only, then per-problem losses are scored on A and B
independently. A family's preference is the mean loss vector over its members,
centred across permutations so it measures preference SHAPE rather than overall
difficulty. If a family has a real preference, its A-vector and its B-vector --
computed from disjoint problems -- must agree.

Three reference points are reported with every number:

    observed     A vs B agreement for the true family labels
    label_null   the same, after shuffling family labels among eval problems.
                 This is the floor: it preserves every marginal and destroys only
                 the family structure.
    chance       1/24 for argmin agreement.

Usage
-----
    python scripts/trajectory_heterogeneity.py \\
        --cache artifacts/depth_cache_k4_dev2400.pt \\
        --config configs/g3_s29_state_ordered.yaml \\
        --out artifacts/c2_trajectory_heterogeneity.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dast.config import ARM_SPECS, load_config  # noqa: E402
from dast.data import load_records  # noqa: E402
from dast.depth_cache import DepthCache  # noqa: E402
from dast.depth_probe import per_problem_permutation_losses  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from permutation_landscape import build_targets  # noqa: E402

MIN_FAMILY_SIZE = 8


def pearson(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first - first.mean()
    second = second - second.mean()
    denominator = float(first.norm() * second.norm())
    return float((first * second).sum() / denominator) if denominator > 1e-12 else 0.0


def fisher_mean(values: list[float]) -> float:
    """Average correlations in z space, which is the correct scale for them."""

    usable = [max(min(value, 0.999999), -0.999999) for value in values]
    if not usable:
        return 0.0
    return math.tanh(sum(math.atanh(value) for value in usable) / len(usable))


def _family_vectors(
    losses: torch.Tensor, families: list[str], keys: list[str]
) -> dict[str, torch.Tensor]:
    """Mean loss vector per family, centred across permutations."""

    buckets: dict[str, list[int]] = defaultdict(list)
    for position, key in enumerate(families):
        buckets[key].append(position)
    vectors = {}
    for key in keys:
        rows = buckets.get(key, [])
        if len(rows) < MIN_FAMILY_SIZE:
            continue
        mean = losses[rows].mean(dim=0)
        vectors[key] = mean - mean.mean()
    return vectors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    config = load_config(args.config)
    cache = DepthCache.load(args.cache)
    stages = cache.num_stages
    targets, distinct = build_targets(config, cache, stages)

    keep = torch.tensor(distinct, dtype=torch.bool)
    hidden = cache.hidden[keep]
    targets = targets[keep]
    kept = [index for index, flag in enumerate(distinct) if flag]
    print(f"cache {tuple(cache.hidden.shape)} layers {cache.layers}; "
          f"kept {len(kept)}/{cache.num_problems} with {stages} distinct targets", flush=True)

    # Strata come from the LIVE records, not from cache metadata: a cache built
    # before the trajectory_lengths fix records a constant, which silently collapses
    # the step_count stratum to one degenerate family.
    records = load_records(
        config.data_path_for_split(cache.metadata["split"]),
        split=cache.metadata["split"],
        limit=cache.num_problems,
    )
    kind = ARM_SPECS["state_ordered"].target_kind
    step_counts = [len(record.raw_targets(kind)) for record in records]
    if [record.problem_id for record in records] != list(cache.problem_ids):
        raise SystemExit("record order does not match the cache; refusing to pair them")

    strata = {
        "generator": [cache.generators[i] for i in kept],
        "answer_type": [cache.answer_types[i] for i in kept],
        "step_count": [str(step_counts[i]) for i in kept],
    }
    # `category` is 1:1 with `generator` in this corpus (30 of each, 30 distinct
    # pairs), so reporting both would look like two independent confirmations of
    # one measurement. Only `generator` is carried.
    assert len(set(cache.generators)) == len(set(zip(cache.generators, cache.categories)))
    perms = list(itertools.permutations(range(stages)))
    count = hidden.shape[0]
    generator = torch.Generator()

    observed: dict[str, list[float]] = defaultdict(list)
    null: dict[str, list[float]] = defaultdict(list)
    argmin_observed: dict[str, list[float]] = defaultdict(list)
    argmin_null: dict[str, list[float]] = defaultdict(list)
    problem_agreement: list[float] = []
    pooled_best: list[int] = []
    previous_argmin: dict[int, int] = {}

    for replicate in range(args.replicates):
        generator.manual_seed(args.seed + replicate)
        order = torch.randperm(count, generator=generator)
        half = count // 2
        quarter = (count - half) // 2
        fit_index = order[:half]
        eval_a = order[half : half + quarter]
        eval_b = order[half + quarter :]

        losses_a = per_problem_permutation_losses(
            hidden, targets, fit_index, eval_a, perms, ridge=args.ridge
        )
        losses_b = per_problem_permutation_losses(
            hidden, targets, fit_index, eval_b, perms, ridge=args.ridge
        )
        pooled_best.append(int(torch.cat([losses_a, losses_b]).mean(dim=0).argmin()))

        # Problem-level: does the SAME problem pick the same permutation when the
        # readout is refitted on a different FIT set?
        current = {}
        for local, global_index in enumerate(eval_a.tolist()):
            current[global_index] = int(losses_a[local].argmin())
        for local, global_index in enumerate(eval_b.tolist()):
            current[global_index] = int(losses_b[local].argmin())
        shared = set(current) & set(previous_argmin)
        if shared:
            problem_agreement.append(
                sum(1 for key in shared if current[key] == previous_argmin[key]) / len(shared)
            )
        previous_argmin = current

        shuffled = torch.randperm(count, generator=generator)
        for name, labels in strata.items():
            labels_a = [labels[i] for i in eval_a.tolist()]
            labels_b = [labels[i] for i in eval_b.tolist()]
            keys = sorted(set(labels_a) & set(labels_b))
            vectors_a = _family_vectors(losses_a, labels_a, keys)
            vectors_b = _family_vectors(losses_b, labels_b, keys)
            shared_keys = sorted(set(vectors_a) & set(vectors_b))
            if not shared_keys:
                continue
            observed[name].append(
                fisher_mean([pearson(vectors_a[k], vectors_b[k]) for k in shared_keys])
            )
            argmin_observed[name].append(
                sum(
                    1 for k in shared_keys
                    if int(vectors_a[k].argmin()) == int(vectors_b[k].argmin())
                ) / len(shared_keys)
            )

            fake = [labels[i] for i in shuffled.tolist()]
            fake_a = [fake[i] for i in range(len(eval_a))]
            fake_b = [fake[i] for i in range(len(eval_a), len(eval_a) + len(eval_b))]
            fake_keys = sorted(set(fake_a) & set(fake_b))
            null_a = _family_vectors(losses_a, fake_a, fake_keys)
            null_b = _family_vectors(losses_b, fake_b, fake_keys)
            null_shared = sorted(set(null_a) & set(null_b))
            if null_shared:
                null[name].append(
                    fisher_mean([pearson(null_a[k], null_b[k]) for k in null_shared])
                )
                argmin_null[name].append(
                    sum(
                        1 for k in null_shared
                        if int(null_a[k].argmin()) == int(null_b[k].argmin())
                    ) / len(null_shared)
                )
        print(f"  replicate {replicate + 1}/{args.replicates}", flush=True)

    chance = 1.0 / len(perms)
    print("\n=== C2: split-half reliability of family-level permutation preference ===")
    print(f"  {'stratum':14s} {'r(A,B)':>9} {'r null':>9} {'argmin':>9} "
          f"{'argmin null':>12} {'chance':>8}")
    summary = {}
    for name in strata:
        if not observed[name]:
            continue
        row = {
            "split_half_r": fisher_mean(observed[name]),
            "split_half_r_label_null": fisher_mean(null[name]) if null[name] else None,
            "argmin_agreement": sum(argmin_observed[name]) / len(argmin_observed[name]),
            "argmin_agreement_label_null": (
                sum(argmin_null[name]) / len(argmin_null[name]) if argmin_null[name] else None
            ),
            "chance_argmin_agreement": chance,
        }
        summary[name] = row
        print(f"  {name:14s} {row['split_half_r']:>+9.4f} "
              f"{(row['split_half_r_label_null'] or 0.0):>+9.4f} "
              f"{row['argmin_agreement']:>9.4f} "
              f"{(row['argmin_agreement_label_null'] or 0.0):>12.4f} {chance:>8.4f}")

    problem_level = sum(problem_agreement) / len(problem_agreement) if problem_agreement else None
    print(f"\n  per-problem argmin agreement across refits: "
          f"{problem_level:.4f} (chance {chance:.4f})" if problem_level is not None else "")
    modal = max(set(pooled_best), key=pooled_best.count)
    print(f"  pooled best permutation: {perms[modal]} in "
          f"{pooled_best.count(modal)}/{len(pooled_best)} replicates")

    payload = {
        "num_problems": int(count),
        "num_stages": stages,
        "num_permutations": len(perms),
        "replicates": args.replicates,
        "min_family_size": MIN_FAMILY_SIZE,
        "strata": summary,
        "per_problem_argmin_agreement": problem_level,
        "chance_argmin_agreement": chance,
        "pooled_best_permutation": list(perms[modal]),
        "pooled_best_stability": pooled_best.count(modal) / len(pooled_best),
        "cache_metadata": cache.metadata,
        "claim_boundary": (
            "Decodability only. A positive split-half reliability would motivate testing "
            "Adaptive DAST; it would not by itself be a capability result. A null forbids "
            "building a router."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
