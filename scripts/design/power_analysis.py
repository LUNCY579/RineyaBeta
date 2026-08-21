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
import math

import numpy as np

# Critical values are computed exactly by t_crit() below. The hard-coded table
# that used to live here only covered the unadjusted alpha = 0.05 case and was
# scaled by a normal-quantile ratio for multiplicity, which was wrong.


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def t_sf2(t: float, df: int) -> float:
    """Two-sided tail P(|T| > t) for Student t with `df` degrees of freedom."""
    if t <= 0:
        return 1.0
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def t_crit(df: int, n_comparisons: int = 1) -> float:
    """Two-sided critical value at alpha = 0.05, Bonferroni-adjusted over a family.

    Computed by inverting the exact t distribution, because the obvious shortcut
    is wrong in the direction that matters. This function used to scale the
    unadjusted t value by a ratio of NORMAL quantiles (zc * t / 1.96), which is
    valid only as df -> infinity and whose error grows as 1/df -- worst at
    exactly the df a k=3 seed design lives at. Measured on 400k null
    simulations, that gave alpha = 0.0340 against a 0.0167 target at df=2, m=3,
    i.e. a family-wise error of 0.098 for a three-member family, roughly double
    the declared 0.05, and every training-level MDE derived from it was 7-43%
    too small.

    scipy is genuinely absent from the environment lock, hence the stdlib
    implementation rather than scipy.stats.t.ppf.
    """
    alpha = 0.05 / max(1, n_comparisons)
    if df >= 10 ** 6:                       # normal limit
        lo, hi = 0.0, 40.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if math.erfc(mid / math.sqrt(2.0)) > alpha:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    lo, hi = 0.0, 400.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_sf2(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


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
