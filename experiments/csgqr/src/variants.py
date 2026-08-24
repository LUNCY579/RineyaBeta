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
    "b9-entropy-whole": Arm(
        "b9-entropy-whole", "B9 entropy gate, whole remnant",
        "confidence alternative: is the failure specific to max_prob?",
        INT4, "confidence_whole", feature="neg_entropy", tau_skip=0.9),
    "b10-margin-whole": Arm(
        "b10-margin-whole", "B10 probability-margin gate, whole remnant",
        "confidence alternative", INT4, "confidence_whole",
        feature="prob_margin", tau_skip=0.9),
    "b11-logitmargin-whole": Arm(
        "b11-logitmargin-whole", "B11 logit-margin gate, whole remnant",
        "confidence alternative; note logit_margin is not confined to [0,1] so "
        "its threshold is in different units", INT4, "confidence_whole",
        feature="logit_margin", tau_skip=4.0),
    "b7-conf-whole-95": Arm(
        "b7-conf-whole-95", "B7b confidence-only whole-remnant gate, tau=0.95",
        "a stricter confidence threshold", INT4, "confidence_whole",
        tau_skip=0.95),
    "b8-conf-random": Arm(
        "b8-conf-random", "B8 confidence + random groups",
        "isolates the confidence contribution from the significance one",
        INT4, "three_level", selector="random"),
    "b10-csgqr": Arm(
        "b10-csgqr", "B10 CSGQR with a STATIC ranking",
        "confidence-gated budget over a fixed global ranking; NOT the dynamic "
        "form -- see b16-csgqr-dyn", INT4, "three_level", selector="static_topk"),
    "b13-wrong-layer": Arm(
        "b13-wrong-layer", "B19 wrong-layer donor, always on",
        "does preserved donor identity matter at a matched byte budget?",
        INT4, "all", donor_offset=6),

    # ----------------------------------------------------------------------
    # Arms that require a trained probe. Until train_router.py produced an
    # artifact these were unrunnable, and the campaign's dynamic-significance
    # and probe-gate findings were offline analyses of traces rather than
    # measurements of a model. Each of these needs --router and/or --probe;
    # CSGQRMlp refuses to construct them without one rather than silently
    # falling back to the static ranking, which is what it used to do.
    # ----------------------------------------------------------------------
    "b9-sig-dyn-25": Arm(
        "b9-sig-dyn-25", "B14 dynamic per-state significance at rho=0.25",
        "significance routing with NO confidence gate: isolates the "
        "'which groups' decision at a matched budget", INT4, "fixed_fraction",
        selector="significance", fraction=0.25),
    "b9-sig-dyn-50": Arm(
        "b9-sig-dyn-50", "B14b dynamic per-state significance at rho=0.50",
        "the same contrast at a wider budget", INT4, "fixed_fraction",
        selector="significance", fraction=0.50),
    "b12-probe-whole": Arm(
        "b12-probe-whole", "B13 hidden-state need probe, whole-remnant gate",
        "the deployment gate: replaces the |V|/(3 d_ff) vocabulary projection "
        "with a d_model linear readout", INT4, "confidence_whole",
        feature="probe_need", tau_skip=0.5),
    "b16-csgqr-dyn": Arm(
        "b16-csgqr-dyn", "B16 full CSGQR: probe gate + dynamic significance",
        "the proposed method in the only form the evidence supports -- probe "
        "for how much, per-state utility for which groups",
        INT4, "three_level", feature="probe_need", selector="significance",
        tau_skip=0.7, tau_full=0.3),
    "b16-csgqr-lens-dyn": Arm(
        "b16-csgqr-lens-dyn", "B16b CSGQR with the exact lens gate",
        "same dynamic significance, but gated on the exact vocabulary lens: "
        "isolates the gate source from the group selector",
        INT4, "three_level", feature="max_prob", selector="significance"),

    # ----------------------------------------------------------------------
    # The matched control for the "how much" gate. Skips a fixed fraction of
    # tokens chosen at RANDOM, whole-remnant. Without this, comparing a gate at
    # 79% activation against always-on at 100% conflates the gate's skill with
    # the compute it spent, and H2 is untestable. tau_skip = t skips 1 - t of
    # tokens, so these are set to match the measured activation of the gate arms
    # they partner: 0.87, 0.84, 0.79, 0.68, 0.65.
    # ----------------------------------------------------------------------
    "r-tok-87": Arm(
        "r-tok-87", "Random token gate at 87% activation",
        "matched control for b11-logitmargin-whole", INT4,
        "confidence_whole", feature="random_token", tau_skip=0.8702),
    "r-tok-84": Arm(
        "r-tok-84", "Random token gate at 84% activation",
        "matched control for b10-margin-whole", INT4,
        "confidence_whole", feature="random_token", tau_skip=0.8356),
    "r-tok-79": Arm(
        "r-tok-79", "Random token gate at 79% activation",
        "matched control for b7-conf-whole", INT4,
        "confidence_whole", feature="random_token", tau_skip=0.7889),
    "r-tok-68": Arm(
        "r-tok-68", "Random token gate at 68% activation",
        "matched control for b12-probe-whole", INT4,
        "confidence_whole", feature="random_token", tau_skip=0.6772),
    "r-tok-65": Arm(
        "r-tok-65", "Random token gate at 65% activation",
        "matched control for b9-entropy-whole", INT4,
        "confidence_whole", feature="random_token", tau_skip=0.6494),
}

# ----------------------------------------------------------------------
# Threshold ladders, so the lens gate and the probe gate can be compared as
# FRONTIERS rather than at one arbitrary operating point each. Comparing a
# max_prob gate at tau=0.9 against a probe gate at tau=0.5 compares two points
# on two curves and says nothing about which curve is better.
# ----------------------------------------------------------------------
for _t in (0.50, 0.70, 0.80, 0.90, 0.95, 0.99):
    CATALOGUE["lens-t" + format(_t, ".2f")] = Arm(
        "lens-t" + format(_t, ".2f"),
        "max_prob whole-remnant gate, tau=" + format(_t, ".2f"),
        "one point on the exact-lens gate frontier", INT4,
        "confidence_whole", feature="max_prob", tau_skip=_t)
for _t in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
    CATALOGUE["probe-t" + format(_t, ".2f")] = Arm(
        "probe-t" + format(_t, ".2f"),
        "probe_need whole-remnant gate, tau=" + format(_t, ".2f"),
        "one point on the hidden-state-probe gate frontier", INT4,
        "confidence_whole", feature="probe_need", tau_skip=_t)
for _t in (0.60, 0.75, 0.90, 0.95):
    CATALOGUE["rtok-t" + format(_t, ".2f")] = Arm(
        "rtok-t" + format(_t, ".2f"),
        "random token gate, tau=" + format(_t, ".2f"),
        "one point on the no-skill matched-compute reference", INT4,
        "confidence_whole", feature="random_token", tau_skip=_t)

LENS_LADDER = tuple("lens-t" + format(t, ".2f")
                    for t in (0.50, 0.70, 0.80, 0.90, 0.95, 0.99))
PROBE_LADDER = tuple("probe-t" + format(t, ".2f")
                     for t in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80))
RTOK_LADDER = tuple("rtok-t" + format(t, ".2f")
                    for t in (0.60, 0.75, 0.90, 0.95))

#: Random-token-gate controls, and the arm each one is matched to.
MATCHED_TOKEN_GATES = {
    "r-tok-87": "b11-logitmargin-whole",
    "r-tok-84": "b10-margin-whole",
    "r-tok-79": "b7-conf-whole",
    "r-tok-68": "b12-probe-whole",
    "r-tok-65": "b9-entropy-whole",
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
         "b6-reversed-25", "b8-conf-random", "b13-wrong-layer")

#: Arms that cannot run without a trained artifact from ``train_router.py``.
#: Listed explicitly so a sweep driver can fail early with a useful message
#: instead of at install time, and so it is impossible to lose track of which
#: arms are the dynamic ones.
NEEDS_ROUTER = ("b9-sig-dyn-25", "b9-sig-dyn-50", "b16-csgqr-dyn",
                "b16-csgqr-lens-dyn")
NEEDS_PROBE = ("b12-probe-whole", "b16-csgqr-dyn")

#: The dynamic half of the method, which was analysed offline but never executed
#: before the probes existed.
TIER3_DYNAMIC = ("b9-sig-dyn-25", "b9-sig-dyn-50", "b12-probe-whole",
                 "b16-csgqr-dyn", "b16-csgqr-lens-dyn")

#: Confidence-feature alternatives, to establish whether the raw-lens failure is
#: specific to max_prob or general to the vocabulary projection.
TIER2_FEATURES = ("b9-entropy-whole", "b10-margin-whole",
                  "b11-logitmargin-whole", "b7-conf-whole-95")
