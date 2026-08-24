#!/usr/bin/env python3
"""Statistics for the CSGQR analyses, with clustering treated as a first-class
concern rather than an afterthought.

Why clustering matters here
---------------------------
Every per-token measurement in this study comes from a prompt that contributed
several tokens. Tokens within a prompt share a context, a topic and a truncated
prefix, so they are not independent draws. A confidence interval computed over
tokens as if they were independent is too narrow -- often by a large factor -- and
would let a null result look significant.

So the bootstrap here resamples **prompts** (clusters), carrying all of a
prompt's tokens with it. Where a token-level interval is also reported it is
labelled as such and is not used for inference.

Everything is implemented directly rather than pulled from scipy, so that
results are reproducible on a machine without it and so that the tie handling
and the discreteness of the exact tests are visible in the code.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch
from torch import Tensor

# Seed for every resampling procedure in this module unless overridden, so a
# reported interval is reproducible from the committed summary alone.
DEFAULT_SEED = 20260823


# --------------------------------------------------------------------------
# Correlation
# --------------------------------------------------------------------------
def _avg_rank(v: Tensor) -> Tensor:
    n = v.numel()
    order = torch.argsort(v)
    ranks = torch.empty(n, dtype=torch.float64)
    ranks[order] = torch.arange(1, n + 1, dtype=torch.float64)
    sorted_v = v[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def pearson(x: Tensor, y: Tensor) -> float:
    a = x.to(torch.float64).flatten()
    b = y.to(torch.float64).flatten()
    a = a - a.mean()
    b = b - b.mean()
    d = (a.norm() * b.norm()).item()
    return (a @ b).item() / d if d else float("nan")


def spearman(x: Tensor, y: Tensor) -> float:
    return pearson(_avg_rank(x.to(torch.float64).flatten()),
                   _avg_rank(y.to(torch.float64).flatten()))


# --------------------------------------------------------------------------
# Ranking / detection metrics
# --------------------------------------------------------------------------
def auroc(score: Tensor, label: Tensor) -> float:
    """AUROC via the rank (Mann-Whitney) identity, with tied scores averaged.

    Returns NaN when one class is absent, rather than a misleading 0.5: an
    undefined AUROC and a chance-level AUROC are different findings.
    """
    s = score.to(torch.float64).flatten()
    y = label.to(torch.float64).flatten()
    n_pos = float((y > 0).sum())
    n_neg = float((y <= 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _avg_rank(s)
    return ((r[y > 0].sum().item() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(score: Tensor, label: Tensor) -> float:
    """Average precision: sum of precision at each positive, over positives.

    This is the step-wise (interpolation-free) estimator, which does not flatter
    a model the way trapezoidal interpolation of a PR curve can.
    """
    s = score.to(torch.float64).flatten()
    y = (label.to(torch.float64).flatten() > 0)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = torch.argsort(s, descending=True)
    hits = y[order].to(torch.float64)
    tp = torch.cumsum(hits, dim=0)
    ranks = torch.arange(1, hits.numel() + 1, dtype=torch.float64)
    precision = tp / ranks
    return (precision * hits).sum().item() / n_pos


def base_rate(label: Tensor) -> float:
    y = label.flatten()
    return float((y > 0).sum()) / max(y.numel(), 1)


# --------------------------------------------------------------------------
# Cluster bootstrap
# --------------------------------------------------------------------------
def cluster_bootstrap(values: Tensor, clusters: list[int], statistic,
                      *, n_boot: int = 2000, seed: int = DEFAULT_SEED,
                      alpha: float = 0.05) -> dict[str, float]:
    """Percentile bootstrap CI resampling whole clusters.

    ``statistic`` maps a 1-D tensor of values to a float. ``clusters`` gives each
    value's cluster id (a prompt index here). Clusters are drawn with
    replacement; every token of a drawn prompt comes along.
    """
    by_cluster: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(clusters):
        by_cluster[c].append(i)
    keys = sorted(by_cluster)
    index = [torch.tensor(by_cluster[k], dtype=torch.long) for k in keys]
    v = values.to(torch.float64).flatten()

    point = statistic(v)
    g = torch.Generator().manual_seed(seed)
    draws = torch.empty(n_boot, dtype=torch.float64)
    n_c = len(keys)
    for b in range(n_boot):
        pick = torch.randint(0, n_c, (n_c,), generator=g)
        sel = torch.cat([index[i] for i in pick.tolist()])
        draws[b] = statistic(v[sel])
    finite = draws[torch.isfinite(draws)]
    if finite.numel() < 2:
        return {"point": point, "lo": float("nan"), "hi": float("nan"),
                "n_clusters": n_c, "n": int(v.numel()), "n_boot": n_boot}
    lo = torch.quantile(finite, alpha / 2).item()
    hi = torch.quantile(finite, 1 - alpha / 2).item()
    return {"point": point, "lo": lo, "hi": hi, "se": finite.std().item(),
            "n_clusters": n_c, "n": int(v.numel()), "n_boot": n_boot}


def _cluster_totals(values: Tensor, clusters: list[int]) -> tuple[Tensor, Tensor]:
    """Per-cluster sums and counts, in cluster-id order."""
    c = torch.as_tensor(clusters, dtype=torch.long)
    n_c = int(c.max().item()) + 1 if c.numel() else 0
    v = values.to(torch.float64).flatten()
    sums = torch.zeros(n_c, dtype=torch.float64).index_add_(0, c, v)
    counts = torch.zeros(n_c, dtype=torch.float64).index_add_(
        0, c, torch.ones_like(v))
    keep = counts > 0
    return sums[keep], counts[keep]


def cluster_bootstrap_mean(values: Tensor, clusters: list[int], *,
                           n_boot: int = 4000, seed: int = DEFAULT_SEED,
                           alpha: float = 0.05) -> dict[str, float]:
    """Prompt-clustered bootstrap CI for a **mean**, vectorized and exact.

    Identical in distribution to :func:`cluster_bootstrap` with
    ``statistic=mean``, but computed in closed form instead of by resampling
    index tensors. The mean of a cluster-resample is

        sum over drawn clusters of (cluster sum) / sum of (cluster sizes)

    so only per-cluster sums and counts are needed. One gather over an
    ``[n_boot, n_clusters]`` index matrix replaces ``n_boot`` Python iterations,
    each of which was concatenating one index tensor per cluster.

    This is not a micro-optimisation. At 416 prompts and 4000 draws the general
    form performs 1.66M small tensor operations *per metric per arm*; across a
    16-arm sweep with five metrics that dominated the wall time of a run whose
    GPU work was a few minutes. Exactness is asserted against the general
    implementation in ``verify_csgqr.py``.
    """
    sums, counts = _cluster_totals(values, clusters)
    n_c = int(sums.numel())
    point = (sums.sum() / counts.sum()).item() if n_c else float("nan")
    if n_c < 2:
        return {"point": point, "lo": float("nan"), "hi": float("nan"),
                "n_clusters": n_c, "n": int(values.numel()), "n_boot": n_boot}
    g = torch.Generator().manual_seed(seed)
    pick = torch.randint(0, n_c, (n_boot, n_c), generator=g)
    draws = sums[pick].sum(dim=1) / counts[pick].sum(dim=1)
    finite = draws[torch.isfinite(draws)]
    return {"point": point,
            "lo": torch.quantile(finite, alpha / 2).item(),
            "hi": torch.quantile(finite, 1 - alpha / 2).item(),
            "se": finite.std().item(),
            "n_clusters": n_c, "n": int(values.numel()), "n_boot": n_boot}


def cluster_bootstrap_mean_pair(a: Tensor, b: Tensor, clusters: list[int], *,
                                n_boot: int = 4000, seed: int = DEFAULT_SEED,
                                alpha: float = 0.05) -> dict[str, float]:
    """Clustered CI for ``mean(a) - mean(b)`` on paired observations.

    Paired, so the difference is formed **per position** before any resampling.
    Resampling ``a`` and ``b`` independently would discard the pairing and
    inflate the interval by roughly the ratio of between- to within-position
    variance -- which for two arms that agree on most tokens is large.
    """
    d = (a.to(torch.float64).flatten() - b.to(torch.float64).flatten())
    out = cluster_bootstrap_mean(d, clusters, n_boot=n_boot, seed=seed,
                                 alpha=alpha)
    out["excludes_zero"] = bool(out["lo"] > 0 or out["hi"] < 0)
    sd = d.std().item()
    out["cohens_d"] = (out["point"] / sd) if sd > 0 else float("nan")
    return out


def cluster_bootstrap_pair(a: Tensor, b: Tensor, clusters: list[int],
                           *, n_boot: int = 2000, seed: int = DEFAULT_SEED,
                           alpha: float = 0.05) -> dict[str, float]:
    """CI for ``mean(a) - mean(b)`` on paired observations, clustered by prompt.

    Paired: the same token contributes to both arms, so the difference is taken
    per token and then resampled. This is strictly tighter than treating the arms
    as independent samples, and it is the correct design when both arms saw the
    same prompts in the same order -- which every arm in this study does.
    """
    d = (a.to(torch.float64).flatten() - b.to(torch.float64).flatten())
    out = cluster_bootstrap(d, clusters, lambda v: v.mean().item(),
                            n_boot=n_boot, seed=seed, alpha=alpha)
    # A two-sided bootstrap p-value: the smallest alpha at which 0 is excluded,
    # approximated by the proportion of draws on the far side of zero.
    out["excludes_zero"] = bool(out["lo"] > 0 or out["hi"] < 0)
    out["cohens_d"] = (d.mean() / d.std()).item() if d.std() > 0 else float("nan")
    return out


# --------------------------------------------------------------------------
# Exact / rank tests
# --------------------------------------------------------------------------
def _log_comb(n: int, k: int) -> float:
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def binomial_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value by the method of small likelihoods."""
    if n == 0:
        return float("nan")
    probs = [math.exp(_log_comb(n, i) + i * math.log(p)
                      + (n - i) * math.log1p(-p)) for i in range(n + 1)]
    target = probs[k]
    return min(1.0, sum(q for q in probs if q <= target * (1 + 1e-12)))


def mcnemar_exact(both: int, only_a: int, only_b: int, neither: int) -> dict:
    """Exact McNemar test on a paired binary 2x2 table.

    ``only_a`` and ``only_b`` are the discordant cells; the concordant cells do
    not enter the p-value, which is why a large N with few discordances still
    yields a weak test. Reported so that is visible.
    """
    n_disc = only_a + only_b
    return {
        "both": both, "only_a": only_a, "only_b": only_b, "neither": neither,
        "n": both + only_a + only_b + neither,
        "n_discordant": n_disc,
        "p_value": binomial_two_sided(only_a, n_disc) if n_disc else float("nan"),
        "odds_direction": ("a" if only_a > only_b
                           else ("b" if only_b > only_a else "tie")),
    }


def wilcoxon_signed_rank(diff: Tensor) -> dict:
    """Wilcoxon signed-rank test with a normal approximation and tie correction.

    Exact enumeration is infeasible at these sample sizes; the normal
    approximation with a continuity correction is used and ``n_nonzero`` is
    reported so the reader can judge whether it is appropriate.
    """
    d = diff.to(torch.float64).flatten()
    d = d[d != 0]
    n = d.numel()
    if n < 10:
        return {"n_nonzero": int(n), "p_value": float("nan"),
                "z": float("nan"), "note": "too few non-zero differences"}
    r = _avg_rank(d.abs())
    w_pos = r[d > 0].sum().item()
    mean = n * (n + 1) / 4.0
    _, counts = torch.unique(r, return_counts=True)
    tie_term = float((counts.to(torch.float64) ** 3 - counts.to(torch.float64)).sum())
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var <= 0:
        return {"n_nonzero": int(n), "p_value": float("nan"), "z": float("nan")}
    z = (w_pos - mean - 0.5 * math.copysign(1.0, w_pos - mean)) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {"n_nonzero": int(n), "w_plus": w_pos, "z": z, "p_value": p}


def holm_bonferroni(pvalues: dict[str, float], alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down correction.

    Used rather than plain Bonferroni because this study runs many related
    comparisons (four confidence features x several thresholds) where plain
    Bonferroni would be needlessly conservative, and rather than
    Benjamini-Hochberg because the claims are individual rather than a
    discovery set.
    """
    items = sorted(((k, v) for k, v in pvalues.items()
                    if v is not None and not math.isnan(v)), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        thresh = alpha / (m - i)
        running = max(running, p * (m - i))
        out[k] = {"p_value": p, "threshold": thresh,
                  "adjusted_p": min(1.0, running),
                  "reject": bool(min(1.0, running) <= alpha)}
    for k, v in pvalues.items():
        if k not in out:
            out[k] = {"p_value": v, "threshold": float("nan"),
                      "adjusted_p": float("nan"), "reject": False}
    return {"alpha": alpha, "n_tests": m, "results": out}


def bin_by(values: Tensor, target: Tensor, edges: list[float]) -> list[dict]:
    """Bucket ``target`` by ``values`` into the given edges."""
    v = values.to(torch.float64).flatten()
    t = target.to(torch.float64).flatten()
    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sel = (v >= lo) & (v < hi) if i < len(edges) - 2 else (v >= lo) & (v <= hi)
        k = int(sel.sum())
        if not k:
            out.append({"lo": lo, "hi": hi, "n": 0})
            continue
        tv = t[sel]
        out.append({
            "lo": lo, "hi": hi, "n": k,
            "mean_value": v[sel].mean().item(),
            "mean_target": tv.mean().item(),
            "median_target": tv.median().item(),
            "frac_target_positive": (tv > 0).double().mean().item(),
        })
    return out
