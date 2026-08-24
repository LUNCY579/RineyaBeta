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

# Which scalar drives the "how much" decision. The first four are read from the
# exact vocabulary lens. ``probe_need`` is read from a trained hidden-state
# probe and is converted to the same "maturity" orientation (higher = less
# remnant needed) before it reaches any threshold, so one policy implementation
# serves both and a tau means the same kind of thing in each.
FEATURES = ("max_prob", "neg_entropy", "prob_margin", "logit_margin",
            "probe_need", "random_token")

# Features that come from the trained probe rather than the vocabulary lens.
# Listing them here rather than testing a string at each call site keeps the
# "does this arm need the LM-head projection?" question answerable in one place,
# which is what decides whether an arm pays the |V|/(3*d_ff) monitor cost.
PROBE_FEATURES = ("probe_need",)

# Features that need no monitor at all. ``random_token`` draws a per-token
# uniform, so ``confidence_whole`` with ``tau_skip = t`` skips a ``1 - t``
# fraction of tokens chosen at random.
#
# This is the matched control the "how much" gate must beat, and without it H2
# ("confidence gating beats random activation at matched compute") is untestable:
# comparing a gate at 79% activation against always-on at 100% conflates the
# gate's *skill* with the compute it spent. Every group-level arm already had a
# random partner; the token-level gate did not.
SYNTHETIC_FEATURES = ("random_token",)


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
        if self.feature not in FEATURES:
            raise ValueError("feature must be one of " + str(FEATURES)
                             + "; got " + str(self.feature))
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

    # -- what this policy requires at runtime ------------------------------
    @property
    def needs_gate_signal(self) -> bool:
        """Does the "how much" decision consult a per-token signal at all?

        ``none``, ``all`` and ``fixed_fraction`` fix ``k`` in advance, so they
        must not pay for any monitor. Keeping this on the policy rather than in
        the module means an arm cannot accidentally pay the projection cost
        without its own configuration saying so.
        """
        return self.mode in ("confidence_whole", "three_level")

    @property
    def needs_lens(self) -> bool:
        """Does it need the exact LM-head vocabulary projection?"""
        return (self.needs_gate_signal
                and self.feature not in PROBE_FEATURES
                and self.feature not in SYNTHETIC_FEATURES)

    @property
    def is_synthetic_gate(self) -> bool:
        """Is the gate signal generated rather than measured?"""
        return self.needs_gate_signal and self.feature in SYNTHETIC_FEATURES

    def synthetic_confidence(self, token_index: int) -> float:
        """A seeded per-token uniform draw, for the ``random_token`` control.

        Keyed on the token index so the arm is reproducible, and on a different
        multiplier from the ``random`` *selector* so a run that uses both does not
        correlate its "whether" and "which" decisions.
        """
        g = torch.Generator().manual_seed(self.seed * 2_654_435_761 + token_index)
        return float(torch.rand(1, generator=g).item())

    @property
    def needs_probe(self) -> bool:
        """Does it need the trained hidden-state probe for the gate signal?"""
        return self.needs_gate_signal and self.feature in PROBE_FEATURES

    @property
    def needs_group_utility(self) -> bool:
        """Does group *selection* need a per-state utility vector?

        True only for the dynamic-significance selector, and only in a mode that
        can actually select a proper subset. Every other selector works from the
        static ``ranking``, which is precisely the static-versus-dynamic contrast
        the ablation rests on -- so this must be an explicit property and not an
        accident of whether a utility function happens to have been passed in.

        The mode guard matters. Under ``none`` and ``all``, ``budget()`` returns
        0 or ``n_groups`` and ``select()`` short-circuits before it ever looks at
        the selector, so no utility vector is consulted. Without the guard, the
        dataclass's own default (``selector="significance"``) made a bare
        ``RoutingPolicy(mode="all")`` unconstructible in ``CSGQRMlp`` -- it
        demanded a trained router for an arm that runs every group
        unconditionally. That never fired in the campaign only because
        ``variants.Arm`` defaults to ``static_topk`` and always passes the
        selector explicitly.
        """
        if self.mode in ("none", "all"):
            return False
        return self.selector == "significance"

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
        """Number of groups to activate, ``k_t``.

        The ``- 1e-9`` before the ceiling is not cosmetic. Callers that want an
        exact ``k`` express it as ``fraction = k / n_groups``, and for many
        ``k`` that round-trip overshoots: ``5/24 * 24 == 5.000000000000001`` in
        binary floating point, so a bare ``ceil`` returns 6. The utility-trace
        pass builds every single-group and pair configuration this way, so the
        bug would have silently mislabelled subset utilities for any subset size
        whose fraction does not round-trip exactly. The tolerance is far below
        any meaningful rho resolution (1e-9 of a group) and cannot absorb a real
        fractional budget.
        """
        r = self.rho(confidence)
        if r <= 0.0:
            return 0
        if r >= 1.0:
            return self.n_groups
        return min(self.n_groups, max(1, int(math.ceil(r * self.n_groups - 1e-9))))

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
            # R_g = (1-C)^gamma * U_g / (c_g + eps). The (1-C)^gamma factor is a
            # per-token *scalar*: it multiplies every group's value equally and
            # therefore cannot change which k groups are largest. It is omitted
            # from the ranking score deliberately, for two reasons. It is
            # mathematically inert here, and including it introduced a real
            # degeneracy -- at C = 1.0 every value became exactly 0.0 and
            # argsort silently returned groups 0..k-1, turning the significance
            # selector into the contiguous selector without any error. The factor
            # remains in the specification because it is what sets *how many*
            # groups run, and that job is done by rho()/budget() above.
            flat_u = utility.to(torch.float64).flatten()
            if flat_u.numel() != self.n_groups:
                raise ValueError("per-state utility has " + str(flat_u.numel())
                                 + " entries; expected " + str(self.n_groups)
                                 + ". Checked before the division so the error "
                                 "names the real problem instead of surfacing "
                                 "as a broadcast failure.")
            # Match the utility's device and dtype. The utility vector arrives
            # from whatever produced it -- a trained router on the GPU, or a
            # hand-built CPU tensor in a test -- and a CPU-only cost vector made
            # this a device mismatch that only fired for the dynamic arms, i.e.
            # after every static arm in a sweep had already run.
            costs = torch.as_tensor(self.group_costs, dtype=flat_u.dtype,
                                    device=flat_u.device)
            value = flat_u / (costs + 1e-12)
            # stable=True so equal-utility groups always resolve to ascending
            # group id. Quantization creates exact ties more often than
            # continuous data would, and an unstable sort would make an arm
            # irreproducible for exactly the tokens where the choice is
            # arbitrary.
            order = torch.argsort(-value, stable=True)
            return order[:k].tolist()
        return self.ranking[:k]

    def to_dict(self) -> dict[str, object]:
        """Every field that changes behaviour, so a run manifest fully pins the arm.

        ``group_costs`` is included even though it is uniform on this checkpoint:
        a serialized policy that omits a field which affects selection cannot be
        replayed, and the omission only becomes visible once mixed-size groups
        exist -- which is the worst moment to discover it.
        """
        return {"mode": self.mode, "feature": self.feature,
                "tau_skip": self.tau_skip, "tau_full": self.tau_full,
                "gamma": self.gamma, "selector": self.selector,
                "fraction": self.fraction, "n_groups": self.n_groups,
                "seed": self.seed, "ranking": list(self.ranking),
                "group_costs": list(self.group_costs),
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
        """Routing accounting for the run.

        Arms with no per-token gate signal (deletion, always-on, fixed budget)
        record their confidence as NaN rather than 0.0. Recording 0.0 would put a
        real-looking "mean_confidence 0.000" in their manifest, which reads as a
        measurement of a maximally-uncertain model rather than as the absence of a
        measurement. NaN entries are excluded here and counted separately.
        """
        t = max(self.tokens, 1)
        conf = torch.tensor(self.confidences, dtype=torch.float64) if self.confidences \
            else torch.zeros(0, dtype=torch.float64)
        real = conf[~torch.isnan(conf)] if conf.numel() else conf
        has_signal = real.numel() > 0
        return {
            "tokens": self.tokens,
            "groups_executed": self.groups_executed,
            "active_fraction": self.groups_executed / (t * self.n_groups),
            "invocation_rate": self.invocations / t,
            "full_invocation_rate": self.full_invocations / t,
            "skip_rate": 1.0 - self.invocations / t,
            "gate_signal_tokens": int(real.numel()),
            "mean_confidence": real.mean().item() if has_signal else None,
            "median_confidence": real.median().item() if has_signal else None,
            "mean_groups_per_token": self.groups_executed / t,
            "group_selection_rate": (self.per_group / t).tolist(),
        }
