#!/usr/bin/env python3
"""Power and minimum detectable effect for the KV x length x seed design.

There are two kinds of comparison in this program and they have different units
of replication. Conflating them is the fastest way to run an underpowered study
that looks well powered.

  INFERENCE-LEVEL contrasts vary only the decode-time configuration -- YaRN
  factor, DCA on/off, KV dtype, evaluation length -- on one fixed checkpoint.
  There is no training run to vary, so the only noise is item sampling, items
  are perfectly paired across arms, and power is governed by item count. These
  are cheap and very well powered.

  TRAINING-LEVEL contrasts compare checkpoints produced by *different training
  runs* (8KV sham vs 4KV vs 2KV vs 1KV). The unit of replication is the run,
  not the item. Adding evaluation items shrinks the within-run term and does
  nothing to the between-run term, so past a few hundred items the seed count
  alone sets the detectable effect. This is the binding constraint on the whole
  program and it is why M1 (seed-variance floor) has to run first: sigma_run is
  the one number that decides whether three seeds is enough.

Everything is Monte Carlo on the actual test that will be used, so there is no
appeal to a closed form that the analysis will not follow. numpy only -- scipy
is not in the environment lock.

Usage:
    python scripts/design/power_analysis.py
    python scripts/design/power_analysis.py --sigma-run 1.5 --comparisons 6
"""

from __future__ import annotations

import argparse

import numpy as np

# Two-sided t critical values at alpha = 0.05, df 1..30; normal beyond.
T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def t_crit(df: int, n_comparisons: int = 1) -> float:
    """Two-sided 5% critical value, Bonferroni-adjusted across a family.

    Holm is uniformly more powerful and is what the analysis will use; Bonferroni
    here makes the reported MDE conservative, which is the right direction for a
    planning number.
    """
    if n_comparisons > 1:
        # Sidak-equivalent df inflation is not tabulated; step up conservatively
        # by widening alpha/m through the normal approximation for df > 30 and
        # by a scaled table lookup otherwise.
        z = {1: 1.960, 2: 2.241, 3: 2.394, 4: 2.498, 5: 2.576, 6: 2.638,
             8: 2.734, 10: 2.807, 12: 2.865, 16: 2.955, 20: 3.023,
             24: 3.078, 30: 3.144, 40: 3.230}
        keys = sorted(z)
        zc = z[min(keys, key=lambda k: abs(k - n_comparisons))]
        base = T_CRIT_95.get(df, 1.960)
        return zc * base / 1.960
    return T_CRIT_95.get(df, 1.960)


def run_level_power(
    delta_pp: float,
    k_seeds: int,
    sigma_run_pp: float,
    n_items: int,
    p_base: float,
    rho_item: float,
    seed_paired: bool,
    n_comparisons: int,
    n_sim: int,
    rng: np.random.Generator,
    cancel: float = 0.5,
) -> float:
    """Power for a training-level contrast, simulating the test that will be run.

    Matched-seed pairing means arm A and arm B are trained at the *same* seeds,
    so they see the same batches in the same order. Because every arm starts
    from one fixed checkpoint there is no weight-init noise to vary; `cancel` is
    the share of between-run *variance* that is common to both arms and so
    cancels in the difference. Pairing then buys variance but pays df: k-1
    instead of 2(k-1), and at k=3 that is 2 df against 4, a critical value of
    4.303 against 2.776. Whether the trade is worth taking depends entirely on
    sigma_run, which is exactly what M1 measures.
    """
    # Within-run sampling SD of one accuracy estimate, percentage points.
    within_sd_pp = 100.0 * np.sqrt(p_base * (1.0 - p_base) / n_items)
    run_sd = sigma_run_pp

    if seed_paired:
        # Per-seed paired difference: shared run component cancels, and item
        # pairing leaves only the (1 - rho) share of the binomial noise.
        resid = np.sqrt(2.0 * (1.0 - cancel)) * run_sd
        item_term = np.sqrt(2.0 * (1.0 - rho_item)) * within_sd_pp
        diff_sd = np.hypot(resid, item_term)
        diffs = rng.normal(delta_pp, diff_sd, size=(n_sim, k_seeds))
        t_stat = diffs.mean(axis=1) / (diffs.std(axis=1, ddof=1) / np.sqrt(k_seeds))
        df = k_seeds - 1
    else:
        # Independent seeds per arm: two-sample t with a pooled variance.
        obs_sd = np.hypot(run_sd, within_sd_pp)
        a = rng.normal(0.0, obs_sd, size=(n_sim, k_seeds))
        b = rng.normal(delta_pp, obs_sd, size=(n_sim, k_seeds))
        va = a.var(axis=1, ddof=1)
        vb = b.var(axis=1, ddof=1)
        s_pool = np.sqrt(0.5 * (va + vb))
        t_stat = (b.mean(axis=1) - a.mean(axis=1)) / (s_pool * np.sqrt(2.0 / k_seeds))
        df = 2 * (k_seeds - 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_abs = np.abs(np.where(np.isfinite(t_stat), t_stat, 0.0))
    return float((t_abs > t_crit(df, n_comparisons)).mean())


def mde(target_power: float = 0.80, lo: float = 0.01, hi: float = 40.0, **kw) -> float:
    """Smallest effect (percentage points) reaching `target_power`, by bisection."""
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        if run_level_power(delta_pp=mid, **kw) < target_power:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def item_level_mde(
    n_items: int, p_base: float, rho_item: float, n_comparisons: int = 1,
    target_power: float = 0.80,
) -> float:
    """MDE for an inference-level contrast: one checkpoint, items paired.

    Normal approximation on the paired difference; with n in the hundreds and
    accuracies away from the boundary this tracks the bootstrap the analysis
    will actually run.
    """
    sd_pp = 100.0 * np.sqrt(2.0 * p_base * (1 - p_base) * (1 - rho_item) / n_items)
    z_a = t_crit(10 ** 6, n_comparisons)
    z_b = 0.842  # 80% power
    return (z_a + z_b) * sd_pp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-sim", type=int, default=20000)
    ap.add_argument("--p-base", type=float, default=0.75, help="baseline accuracy on the endpoint")
    ap.add_argument("--rho-item", type=float, default=0.6,
                    help="across-arm correlation from shared item difficulty")
    ap.add_argument("--comparisons", type=int, default=3,
                    help="size of the multiplicity family for the primary endpoint")
    ap.add_argument("--sigma-run", type=float, default=None,
                    help="if set, report a single scenario at this between-run SD (pp)")
    ap.add_argument("--cancel", type=float, default=0.5,
                    help="share of between-run variance shared across arms at a matched seed")
    ap.add_argument("--seed", type=int, default=20260819)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    common = dict(n_items=400, p_base=args.p_base, rho_item=args.rho_item,
                  n_comparisons=args.comparisons, n_sim=args.n_sim, rng=rng,
                  cancel=args.cancel)

    print("INFERENCE-LEVEL contrasts (one checkpoint; YaRN factor, DCA, KV dtype, length)")
    print("no training variance; unit of replication = item\n")
    print(f"{'items':>7} {'MDE @ m=1':>11} {'MDE @ m=3':>11} {'MDE @ m=12':>11}")
    for n in (100, 200, 400, 800, 1600):
        row = [item_level_mde(n, args.p_base, args.rho_item, m) for m in (1, 3, 12)]
        print(f"{n:>7} " + "".join(f"{v:>10.2f}pp" for v in row))
    print(f"\n(baseline accuracy {args.p_base:.2f}, item correlation across arms {args.rho_item:.2f})\n")

    print("=" * 78)
    print("TRAINING-LEVEL contrasts (8KV sham vs 4KV vs 2KV vs 1KV)")
    print("unit of replication = training run; items cannot buy back run variance\n")

    sigmas = [args.sigma_run] if args.sigma_run else [0.5, 1.0, 1.5, 2.0, 3.0]
    ks = (3, 4, 5, 6, 8)

    for paired, label in ((True, "matched-seed pairing"), (False, "independent seeds per arm")):
        print(f"MDE at 80% power, {label}, 400 items, family m={args.comparisons}"
              f"{', cancel=' + format(args.cancel, '.2f') if paired else ''}")
        print(f"{'sigma_run':>10} " + "".join(f"{'k=' + str(k):>10}" for k in ks))
        for s in sigmas:
            cells = [f"{mde(k_seeds=k, sigma_run_pp=s, seed_paired=paired, **common):>8.2f}pp"
                     for k in ks]
            print(f"{s:>9.1f}pp " + "".join(cells))
        print()

    print("items are nearly free past a few hundred; seeds are not.")
    print("sweep of item count at k=3, sigma_run=1.5pp, matched seeds:")
    for n in (100, 200, 400, 800, 1600, 3200):
        kw = dict(common, n_items=n)
        v = mde(k_seeds=3, sigma_run_pp=1.5, seed_paired=True, **kw)
        print(f"  {n:>5} items -> {v:5.2f}pp")

    print("\nfloor imposed by run variance alone (items -> infinity), matched seeds:")
    print(f"{'sigma_run':>10} " + "".join(f"{'k=' + str(k):>10}" for k in ks))
    for s in sigmas:
        kw = dict(common, n_items=10 ** 7)
        cells = [f"{mde(k_seeds=k, sigma_run_pp=s, seed_paired=True, **kw):>8.2f}pp" for k in ks]
        print(f"{s:>9.1f}pp " + "".join(cells))


if __name__ == "__main__":
    main()
