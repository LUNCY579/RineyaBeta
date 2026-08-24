#!/usr/bin/env python3
"""The experimental arms, as data.

Each arm is a (quantization spec, routing policy, donor) triple. Keeping them in
one table rather than scattered across command lines means an arm's definition is
committed, diffable, and identical between the quality run, the latency run and
the memory run -- so a number from one can be placed beside a number from
another.

The controls are deliberately in the same table as the method. ``random``,
``shuffled``, ``reversed``, ``magnitude`` and ``wrong-layer`` are not
afterthoughts to be added if the method looks good; they are the arms that
decide whether it looks good for the right reason. They were written before any
CSGQR result was inspected.

Matched compute
---------------
Several arms exist only to be compared at an *equal active-group fraction*. A
routing arm that activates 40% of groups may only be compared against a random
arm that also activates 40%. ``matched_random`` and ``matched_static`` build
those partners from a measured fraction, which is why they are functions rather
than table entries: the fraction is not known until the confidence-gated arm has
run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from quant import QuantSpec
from router import RoutingPolicy


@dataclass(frozen=True)
class Arm:
    """One experimental configuration."""

    key: str
    label: str
    purpose: str
    spec: QuantSpec
    mode: str
    selector: str = "static_topk"
    feature: str = "max_prob"
    tau_skip: float = 0.9
    tau_full: float = 0.3
    gamma: float = 1.0
    fraction: float = 1.0
    n_groups: int = 24
    donor_offset: int | None = None      # None = correct layer
    dense: bool = False                  # no surgery at all
    ranking: tuple[int, ...] = field(default_factory=tuple)

    def policy(self, *, ranking: list[int] | None = None,
               n_groups: int | None = None) -> RoutingPolicy:
        n = n_groups if n_groups is not None else self.n_groups
        rank = list(ranking if ranking is not None else self.ranking) or list(range(n))
        return RoutingPolicy(mode=self.mode, feature=self.feature,
                             tau_skip=self.tau_skip, tau_full=self.tau_full,
                             gamma=self.gamma, selector=self.selector,
                             fraction=self.fraction, n_groups=n, ranking=rank)


FP = QuantSpec(16, 0, True)
INT8 = QuantSpec(8, 128, False)
INT4 = QuantSpec(4, 128, False)

CATALOGUE: dict[str, Arm] = {
    "b0-dense": Arm(
        "b0-dense", "B0 dense reference", "quality ceiling; the unmodified model",
        FP, "all", dense=True),
    "b1-delete": Arm(
        "b1-delete", "B1 exact MLP deletion", "damage floor",
        INT4, "none", n_groups=1),
    "b2-fp-all": Arm(
        "b2-fp-all", "B2 frozen FP donor, always on",
        "isolates freezing AND the remnant code path; defines the noise floor",
        FP, "all"),
    "b3-int8-all": Arm(
        "b3-int8-all", "B3 always-on INT8 remnant", "high-precision remnant ceiling",
        INT8, "all"),
    "b4-int4-all": Arm(
        "b4-int4-all", "B4 always-on INT4 remnant",
        "low-bit remnant ceiling; the upper bound on any routed INT4 arm",
        INT4, "all"),
    "b5-random-25": Arm(
        "b5-random-25", "B5 random groups at rho=0.25",
        "does intelligent routing matter at matched compute?",
        INT4, "fixed_fraction", selector="random", fraction=0.25),
    "b5-random-50": Arm(
        "b5-random-50", "B5 random groups at rho=0.50",
        "matched-compute control", INT4, "fixed_fraction",
        selector="random", fraction=0.50),
    "b6-static-25": Arm(
        "b6-static-25", "B6 static top-significance at rho=0.25",
        "does a fixed global ranking suffice?", INT4, "fixed_fraction",
        selector="static_topk", fraction=0.25),
    "b6-static-50": Arm(
        "b6-static-50", "B6 static top-significance at rho=0.50",
        "does a fixed global ranking suffice?", INT4, "fixed_fraction",
        selector="static_topk", fraction=0.50),
    "b6-magnitude-25": Arm(
        "b6-magnitude-25", "B6b magnitude-ranked groups at rho=0.25",
        "the cheap heuristic significance must beat", INT4, "fixed_fraction",
        selector="magnitude", fraction=0.25),
    "b6-shuffled-25": Arm(
        "b6-shuffled-25", "B6c shuffled significance at rho=0.25",
        "same groups, destroyed order: does the ORDER carry information?",
        INT4, "fixed_fraction", selector="static_topk", fraction=0.25),
    "b6-reversed-25": Arm(
        "b6-reversed-25", "B6d reversed significance at rho=0.25",
        "worst-first; if this ties significance the signal is noise",
        INT4, "fixed_fraction", selector="static_topk", fraction=0.25),
    "b7-conf-whole": Arm(
        "b7-conf-whole", "B7 confidence-only whole-remnant gate",
        "the simplest confidence method", INT4, "confidence_whole",
        tau_skip=0.9),
    "b8-conf-random": Arm(
        "b8-conf-random", "B8 confidence + random groups",
        "isolates the confidence contribution from the significance one",
        INT4, "three_level", selector="random"),
    "b9-sig-only": Arm(
        "b9-sig-only", "B9 significance routing without confidence",
        "isolates the significance contribution", INT4, "fixed_fraction",
        selector="static_topk", fraction=0.5),
    "b10-csgqr": Arm(
        "b10-csgqr", "B10 full CSGQR", "the proposed method",
        INT4, "three_level", selector="static_topk"),
    "b13-wrong-layer": Arm(
        "b13-wrong-layer", "B13 wrong-layer donor, always on",
        "does preserved donor identity matter at a matched byte budget?",
        INT4, "all", donor_offset=6),
}


def matched_random(fraction: float, spec: QuantSpec = INT4,
                   n_groups: int = 24) -> Arm:
    """A random-selection arm at a measured active fraction."""
    return Arm("matched-random-" + format(fraction, ".3f"),
               "random groups at rho=" + format(fraction, ".3f"),
               "matched-compute control for a confidence-gated arm",
               spec, "fixed_fraction", selector="random", fraction=fraction,
               n_groups=n_groups)


def matched_static(fraction: float, spec: QuantSpec = INT4,
                   n_groups: int = 24) -> Arm:
    """A static-ranking arm at a measured active fraction."""
    return Arm("matched-static-" + format(fraction, ".3f"),
               "static top-significance at rho=" + format(fraction, ".3f"),
               "matched-compute control for a confidence-gated arm",
               spec, "fixed_fraction", selector="static_topk", fraction=fraction,
               n_groups=n_groups)


TIER1 = ("b0-dense", "b1-delete", "b2-fp-all", "b3-int8-all", "b4-int4-all",
         "b5-random-25", "b6-static-25", "b7-conf-whole", "b10-csgqr")
TIER2 = ("b5-random-50", "b6-static-50", "b6-magnitude-25", "b6-shuffled-25",
         "b6-reversed-25", "b8-conf-random", "b9-sig-only", "b13-wrong-layer")
