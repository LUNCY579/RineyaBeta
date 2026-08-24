#!/usr/bin/env python3
"""Routing policies: how much remnant to run, and which groups.

Every experimental arm is a configuration of :class:`RoutingPolicy`, so that a
difference between arms is a difference of policy and never a difference of
implementation. Deletion is ``mode="none"``; an always-on remnant is
``mode="all"``; full CSGQR is ``mode="three_level"``. The same
:meth:`RoutingPolicy.select` returns the group list in all cases, and
``forward_groups`` on the remnant honours it by slicing before it dequantizes.

The three-level policy
----------------------
With ``tau_full < tau_skip`` and confidence ``C``:

* ``C >= tau_skip``  -> no groups. The remnant is bypassed entirely.
* ``C <= tau_full``  -> every group. The full quantized remnant runs.
* in between         -> a significance-selected fraction

  .. code::

      rho = ( (tau_skip - C) / (tau_skip - tau_full) ) ** gamma
      k   = ceil(rho * n_groups)

  so ``rho`` falls from 1 at ``tau_full`` to 0 at ``tau_skip``, and ``gamma``
  bends the ramp: ``gamma < 1`` spends compute more readily, ``gamma > 1`` is
  stingier. ``gamma = 1`` is linear.

Note ``k = ceil(rho * n)`` means any confidence strictly below ``tau_skip``
activates at least one group. That is deliberate -- the interior region should
not silently contain a second skip region -- but it makes ``k=0`` reachable only
through the ``tau_skip`` branch, which is what the activation-rate accounting
assumes.

Cost-aware selection
--------------------
When group costs differ, groups are ranked by

.. code::

    R_g = (1 - C)^gamma * U^_g / (c_g + eps)

and the top ``k`` taken. On this checkpoint every group is the same shape and so
has the same cost, which makes ``R_g`` a monotone rescaling of ``U^_g`` and the
selection identical to ranking on utility alone. The cost term is implemented
anyway because mixed-size groups are a planned follow-up, and because a cost
model that only appears later tends to appear wrong.

What this does *not* do
-----------------------
It does not compute the remnant and multiply by zero. ``select`` returns a list
of group ids; nothing downstream sees the inactive groups at all. The
distinction between mathematical sparsity and executed sparsity is the whole
question for Gate E, so it is enforced here at the interface rather than trusted
to a caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

MODES = ("none", "all", "confidence_whole", "fixed_fraction", "three_level")
SELECTORS = ("significance", "random", "magnitude", "shuffled", "reversed",
             "static_topk", "contiguous")


@dataclass
class RoutingPolicy:
    """A complete, serializable description of one arm's routing behaviour.

    ``feature`` names which confidence scalar drives the gate
    (``max_prob``, ``neg_entropy``, ``prob_margin``, ``logit_margin``). The
    thresholds are in that feature's units, which is why they are swept per
    feature rather than shared.
    """

    mode: str = "three_level"
    feature: str = "max_prob"
    tau_skip: float = 0.9
    tau_full: float = 0.3
    gamma: float = 1.0
    selector: str = "significance"
    fraction: float = 1.0          # used by mode="fixed_fraction"
    n_groups: int = 24
    seed: int = 20260823
    ranking: list[int] = field(default_factory=list)
    group_costs: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError("mode must be one of " + str(MODES) + "; got " + str(self.mode))
        if self.selector not in SELECTORS:
            raise ValueError("selector must be one of " + str(SELECTORS)
                             + "; got " + str(self.selector))
        if self.mode == "three_level" and not self.tau_full < self.tau_skip:
            raise ValueError("three_level needs tau_full < tau_skip; got "
                             + str(self.tau_full) + " >= " + str(self.tau_skip))
        if self.gamma <= 0:
            raise ValueError("gamma must be > 0; got " + str(self.gamma))
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError("fraction must be in [0, 1]; got " + str(self.fraction))
        if not self.ranking:
            self.ranking = list(range(self.n_groups))
        if sorted(self.ranking) != list(range(self.n_groups)):
            raise ValueError("ranking must be a permutation of 0..n_groups-1")
        if not self.group_costs:
            self.group_costs = [1.0] * self.n_groups

    def name(self) -> str:
        if self.mode == "none":
            return "delete"
        if self.mode == "all":
            return "remnant-all"
        if self.mode == "fixed_fraction":
            return ("fixed-" + format(self.fraction, ".3f") + "-" + self.selector)
        if self.mode == "confidence_whole":
            return "confwhole-" + self.feature + "-t" + format(self.tau_skip, ".3f")
        return ("csgqr-" + self.feature + "-s" + format(self.tau_skip, ".3f")
                + "-f" + format(self.tau_full, ".3f") + "-g" + format(self.gamma, ".2f")
                + "-" + self.selector)

    # -- selection ---------------------------------------------------------
    def rho(self, confidence: float) -> float:
        """Target active fraction for one token."""
        if self.mode == "none":
            return 0.0
        if self.mode == "all":
            return 1.0
        if self.mode == "fixed_fraction":
            return self.fraction
        if self.mode == "confidence_whole":
            return 0.0 if confidence >= self.tau_skip else 1.0
        if confidence >= self.tau_skip:
            return 0.0
        if confidence <= self.tau_full:
            return 1.0
        span = self.tau_skip - self.tau_full
        return ((self.tau_skip - confidence) / span) ** self.gamma

    def budget(self, confidence: float) -> int:
        """Number of groups to activate, ``k_t``."""
        r = self.rho(confidence)
        if r <= 0.0:
            return 0
        if r >= 1.0:
            return self.n_groups
        return min(self.n_groups, int(math.ceil(r * self.n_groups)))

    def select(self, confidence: float, *, utility: Tensor | None = None,
               token_index: int = 0) -> list[int]:
        """Group ids to execute for one token.

        ``utility`` is a per-group predicted-utility vector for *this* state,
        used by ``selector="significance"``. When it is absent the policy falls
        back to its static ``ranking``, which is exactly the
        static-versus-dynamic significance contrast the ablation needs -- so the
        fallback is a real arm, not an error path.
        """
        k = self.budget(confidence)
        if k <= 0:
            return []
        if k >= self.n_groups:
            return list(range(self.n_groups))

        if self.selector == "random":
            # Seeded per token so the arm is reproducible but not correlated
            # with group identity across tokens.
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + token_index)
            return torch.randperm(self.n_groups, generator=g)[:k].tolist()
        if self.selector == "contiguous":
            return list(range(k))
        if self.selector == "significance" and utility is not None:
            costs = torch.tensor(self.group_costs, dtype=torch.float64)
            weight = max(1.0 - confidence, 0.0) ** self.gamma
            value = weight * utility.to(torch.float64) / (costs + 1e-12)
            return torch.argsort(value, descending=True)[:k].tolist()
        return self.ranking[:k]

    def to_dict(self) -> dict[str, object]:
        return {"mode": self.mode, "feature": self.feature,
                "tau_skip": self.tau_skip, "tau_full": self.tau_full,
                "gamma": self.gamma, "selector": self.selector,
                "fraction": self.fraction, "n_groups": self.n_groups,
                "seed": self.seed, "ranking": list(self.ranking),
                "name": self.name()}


class RoutingTrace:
    """Per-token routing accounting for one evaluation run.

    ``active_fraction`` is the quantity that decides whether an arm is a real
    adaptive method or an always-on layer wearing a gate. It is the mean of
    ``k_t / n_groups`` over *tokens*, not over prompts, so a long generation
    does not get the same weight as a one-token reply.
    """

    def __init__(self, n_groups: int) -> None:
        self.n_groups = n_groups
        self.tokens = 0
        self.groups_executed = 0
        self.invocations = 0            # tokens with at least one group active
        self.full_invocations = 0       # tokens with every group active
        self.per_group = torch.zeros(n_groups, dtype=torch.float64)
        self.confidences: list[float] = []
        self.budgets: list[int] = []

    def record(self, confidence: float, groups: list[int]) -> None:
        self.tokens += 1
        k = len(groups)
        self.groups_executed += k
        self.invocations += int(k > 0)
        self.full_invocations += int(k == self.n_groups)
        for g in groups:
            self.per_group[g] += 1
        self.confidences.append(float(confidence))
        self.budgets.append(k)

    def summary(self) -> dict[str, object]:
        t = max(self.tokens, 1)
        conf = torch.tensor(self.confidences, dtype=torch.float64) if self.confidences \
            else torch.zeros(1, dtype=torch.float64)
        return {
            "tokens": self.tokens,
            "groups_executed": self.groups_executed,
            "active_fraction": self.groups_executed / (t * self.n_groups),
            "invocation_rate": self.invocations / t,
            "full_invocation_rate": self.full_invocations / t,
            "skip_rate": 1.0 - self.invocations / t,
            "mean_confidence": conf.mean().item(),
            "median_confidence": conf.median().item(),
            "mean_groups_per_token": self.groups_executed / t,
            "group_selection_rate": (self.per_group / t).tolist(),
        }
