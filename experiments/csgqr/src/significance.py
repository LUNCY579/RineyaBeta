#!/usr/bin/env python3
"""Marginal-recovery utility per remnant group, and the rankings built from it.

Definition
----------
For a token ``t``, with ``p^T`` the dense teacher's next-token distribution,
``p^0`` the distribution with the remnant deleted, and ``p^{+g}`` the
distribution with only group ``g`` active::

    U_{t,g} = KL(p^T || p^0) - KL(p^T || p^{+g})

Positive means group ``g`` moved the truncated model *toward* the teacher;
negative means it made things worse. The whole-remnant utility ``U_t`` uses
``p^Q`` (all groups) in place of ``p^{+g}``.

Sign convention matters and is easy to get backwards: this is
``damage_without - damage_with``, so **larger is better** and a useless group
scores ~0 rather than ~1. Every consumer of these numbers assumes that.

Exposure normalisation
----------------------
A group observed more often must not out-rank a group that is more useful. The
accumulator therefore keeps ``score`` and ``exposure`` separately and ranks on
``score / exposure``, mirroring the convention already used by
``scripts/kv/significance.py`` in this repository. When every group is exposed on
every token -- which is the case for the labelling pass, since it evaluates all
groups per token -- the normalisation is a no-op; it earns its keep for
online/sampled accumulation where exposure is uneven by construction.

Causal, not correlational
-------------------------
``U_{t,g}`` is measured by *intervening*: actually running the model with group
``g`` on and comparing against it off. It is not an activation-magnitude proxy.
That distinction is the point of the ``magnitude`` ranking below, which exists
purely as the control that a magnitude heuristic has to beat.
"""

from __future__ import annotations

import torch
from torch import Tensor


def kl_divergence(logits_p: Tensor, logits_q: Tensor) -> Tensor:
    """``KL(p || q)`` in nats, from logits, reduced over the last axis.

    float32 throughout: a bf16 log-softmax over 32768 classes carries enough
    error to move these divergences in the third decimal, and the differences
    being measured here are of that order.
    """
    lp = torch.log_softmax(logits_p.to(torch.float32), dim=-1)
    lq = torch.log_softmax(logits_q.to(torch.float32), dim=-1)
    return (lp.exp() * (lp - lq)).sum(dim=-1)


def marginal_utility(teacher: Tensor, deleted: Tensor, candidate: Tensor) -> Tensor:
    """``KL(p^T||p^0) - KL(p^T||p^cand)``. Larger is better."""
    return kl_divergence(teacher, deleted) - kl_divergence(teacher, candidate)


class GroupSignificance:
    """Accumulates per-group utility with separate exposure counts."""

    def __init__(self, n_groups: int) -> None:
        self.n_groups = n_groups
        self.score = torch.zeros(n_groups, dtype=torch.float64)
        self.exposure = torch.zeros(n_groups, dtype=torch.float64)
        self.positive = torch.zeros(n_groups, dtype=torch.float64)

    def update(self, utilities: Tensor, groups: list[int] | None = None) -> None:
        """Add observations. ``utilities`` is ``[n_observed]`` or ``[N, n_observed]``."""
        u = utilities.to(torch.float64).reshape(-1, utilities.shape[-1])
        idx = list(range(self.n_groups)) if groups is None else groups
        if u.shape[-1] != len(idx):
            raise ValueError("expected " + str(len(idx)) + " group columns; got "
                             + str(u.shape[-1]))
        for col, g in enumerate(idx):
            self.score[g] += u[:, col].sum()
            self.exposure[g] += u.shape[0]
            self.positive[g] += (u[:, col] > 0).sum()

    def mean_utility(self) -> Tensor:
        return self.score / self.exposure.clamp(min=1.0)

    def positive_rate(self) -> Tensor:
        return self.positive / self.exposure.clamp(min=1.0)

    def ranking(self) -> list[int]:
        """Group ids, most useful first."""
        return torch.argsort(self.mean_utility(), descending=True).tolist()

    def to_dict(self) -> dict[str, object]:
        return {
            "n_groups": self.n_groups,
            "mean_utility": self.mean_utility().tolist(),
            "positive_rate": self.positive_rate().tolist(),
            "exposure": self.exposure.tolist(),
            "ranking": self.ranking(),
        }


# --------------------------------------------------------------------------
# Rankings, including the controls that significance has to beat
# --------------------------------------------------------------------------
def magnitude_ranking(remnant) -> list[int]:
    """Groups ranked by weight magnitude -- the cheap heuristic control.

    Uses the Frobenius norm of the group's dequantized ``down`` rows, which is
    what scales the group's contribution to the residual stream. If significance
    cannot beat this, significance is not carrying information beyond "this group
    has big weights".
    """
    scores = []
    for g in range(remnant.n_groups):
        lo, hi = remnant.bounds[g]
        w = remnant.down_q.dequantize(lo, hi).to(torch.float32)
        scores.append(w.norm().item())
    return torch.argsort(torch.tensor(scores), descending=True).tolist()


def shuffled_ranking(ranking: list[int], seed: int) -> list[int]:
    """A fixed permutation of a ranking: same groups, destroyed order.

    The control for "does the *order* carry information", as distinct from
    "does activating this many groups help".
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ranking), generator=g).tolist()
    return [ranking[i] for i in perm]


def reversed_ranking(ranking: list[int]) -> list[int]:
    """Worst-first. If this matches significance, the signal is noise."""
    return list(reversed(ranking))


def rank_correlation(a: Tensor, b: Tensor) -> dict[str, float]:
    """Spearman and Kendall-tau-b between two score vectors.

    Hand-rolled rather than pulled from scipy so the result is identical on a
    machine without scipy, and so that tie handling is visible: Spearman uses
    average ranks for ties, which matters because quantization creates exact
    ties in group utility more often than continuous data would.
    """
    x = a.to(torch.float64).flatten()
    y = b.to(torch.float64).flatten()
    if x.numel() != y.numel():
        raise ValueError("length mismatch")
    n = x.numel()
    if n < 2:
        return {"spearman": float("nan"), "kendall_tau_b": float("nan"), "n": n}

    def avg_rank(v: Tensor) -> Tensor:
        order = torch.argsort(v)
        ranks = torch.empty(n, dtype=torch.float64)
        ranks[order] = torch.arange(1, n + 1, dtype=torch.float64)
        # Average ranks within tied groups.
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

    rx, ry = avg_rank(x), avg_rank(y)
    rxc, ryc = rx - rx.mean(), ry - ry.mean()
    denom = (rxc.norm() * ryc.norm()).item()
    spearman = (rxc @ ryc).item() / denom if denom else float("nan")

    concordant = discordant = tie_x = tie_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = (x[i] - x[j]).item()
            dy = (y[i] - y[j]).item()
            if dx == 0 and dy == 0:
                tie_x += 1
                tie_y += 1
            elif dx == 0:
                tie_x += 1
            elif dy == 0:
                tie_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    pairs = n * (n - 1) // 2
    d = math_sqrt((pairs - tie_x) * (pairs - tie_y))
    tau = (concordant - discordant) / d if d else float("nan")
    return {"spearman": spearman, "kendall_tau_b": tau, "n": n}


def math_sqrt(value: float) -> float:
    import math
    return math.sqrt(max(value, 0.0))
