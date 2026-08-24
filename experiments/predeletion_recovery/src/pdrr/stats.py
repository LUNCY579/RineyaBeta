"""Paired confirmatory statistics with explicit multiplicity control."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ContrastResult:
    name: str
    n_clusters: int
    mean_difference: float
    confidence_low: float
    confidence_high: float
    raw_p: float
    adjusted_p: float | None = None
    practical_pass: bool = False
    statistical_pass: bool = False


def cluster_means(differences: Mapping[str, Sequence[float]]) -> np.ndarray:
    means = [float(np.mean(values)) for _, values in sorted(differences.items())
             if len(values)]
    if not means:
        raise ValueError("no paired clusters")
    return np.asarray(means, dtype=float)


def cluster_bootstrap_ci(
    differences: Mapping[str, Sequence[float]],
    *,
    samples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    means = cluster_means(differences)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for i in range(samples):
        sampled = rng.choice(means, size=len(means), replace=True)
        draws[i] = sampled.mean()
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def sign_flip_pvalue(
    differences: Mapping[str, Sequence[float]],
    *,
    samples: int = 100_000,
    seed: int = 0,
) -> float:
    means = cluster_means(differences)
    observed = abs(float(means.mean()))
    n = len(means)
    if n <= 18:
        values = []
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            values.append(abs(float(np.mean(means * np.asarray(signs)))))
        return float((sum(v >= observed - 1e-15 for v in values)) / len(values))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(samples):
        signs = rng.choice((-1.0, 1.0), size=n)
        extreme += abs(float(np.mean(means * signs))) >= observed - 1e-15
    return (extreme + 1.0) / (samples + 1.0)


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, pvalue) in enumerate(ordered):
        candidate = min(1.0, (m - rank) * float(pvalue))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def paired_contrast(
    name: str,
    differences: Mapping[str, Sequence[float]],
    *,
    practical_threshold: float,
    bootstrap_samples: int = 10_000,
    permutation_samples: int = 100_000,
    seed: int = 0,
) -> ContrastResult:
    means = cluster_means(differences)
    low, high = cluster_bootstrap_ci(
        differences, samples=bootstrap_samples, seed=seed)
    pvalue = sign_flip_pvalue(
        differences, samples=permutation_samples, seed=seed + 1)
    mean = float(means.mean())
    return ContrastResult(
        name=name,
        n_clusters=len(means),
        mean_difference=mean,
        confidence_low=low,
        confidence_high=high,
        raw_p=pvalue,
        practical_pass=mean >= practical_threshold,
        statistical_pass=False,
    )


def finalize_family(results: Sequence[ContrastResult], alpha: float = 0.05) -> list[ContrastResult]:
    adjusted = holm_adjust({result.name: result.raw_p for result in results})
    final = []
    for result in results:
        adj = adjusted[result.name]
        final.append(ContrastResult(
            **{**result.__dict__, "adjusted_p": adj,
               "statistical_pass": adj <= alpha and result.confidence_low > 0.0}
        ))
    return final
