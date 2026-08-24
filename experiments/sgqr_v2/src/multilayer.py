#!/usr/bin/env python3
"""SGQR-v2: retire several transformer MLPs at once, as frozen low-bit remnants.

Why this is a thin layer and not a rewrite
------------------------------------------
Everything hard here was already built and verified for CSGQR-v1: packed INT4
storage with hand-checked nibble decoding, exact additive group decomposition
(< 1e-14 in float64), row-slice dequantization that provably never touches an
inactive group's bytes (proven by instrumentation, not by timing), donor freezing
by construction, and a deployment artifact that reloads without the dense donor.
144 verification checks and 40 independent re-derivations cover those.

So this module does exactly one new thing: it manages a **set** of retired
layers and a **global** budget across them. Reimplementing the remnant would
throw away the evidence.

The one genuinely new design decision
-------------------------------------
With four retired layers there are 4 x 24 = 96 selectable groups, and the budget
has to be split among them. Two allocations are implemented:

``global``   one top-k over all 96 groups, ranked by interventional utility. This
             lets significance decide *where* compute goes, including deciding
             that one layer deserves more of it than another.
``uniform``  the same rho within every layer. The control that asks whether the
             cross-layer part of the allocation is doing anything.

``global`` is the method; ``uniform`` is its control. Reporting only the first
would leave "did it help to move budget between layers?" unanswered.

What is deliberately absent
---------------------------
No vocabulary-head confidence gate: it costs ``|V| / (3 d_ff)`` = 3.56x the layer
it gates in MACs and 13.96x measured, so it cannot pay for itself at any
threshold. No per-token dynamic router: it was genuinely state-conditioned and
did not beat a fixed ranking at either budget tested. No three-level ramp: it tied
a binary gate. All three remain runnable in ``experiments/csgqr/`` as historical
baselines -- they are dropped from the default, not deleted.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CSGQR_SRC = REPO / "experiments" / "csgqr" / "src"
for _p in (str(CSGQR_SRC), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quant import QuantSpec                                        # noqa: E402
from remnant import QuantizedMlpRemnant, RemnantConfig             # noqa: E402


# --------------------------------------------------------------------------
# A retired layer's MLP, replaced by a frozen remnant with a fixed group set
# --------------------------------------------------------------------------
class SGQRMlp(nn.Module):
    """Drop-in replacement for one Qwen3 MLP: run a fixed subset of groups.

    Deliberately simpler than ``CSGQRMlp``. There is no gate, no per-token
    branch, no bucketing: the active group set is fixed for the whole run, so
    every token takes the same path and the module is one slice plus three
    matmuls. That is the "clean scientific core" the campaign asks for, and it
    also removes the per-token Python loop that cost CSGQR-v1 a 16x
    reference-backend penalty on its dynamic arms.
    """

    def __init__(self, remnant: QuantizedMlpRemnant, groups: list[int]) -> None:
        super().__init__()
        self.remnant = remnant
        self.set_groups(groups)

    def set_groups(self, groups: list[int]) -> None:
        g = sorted(set(int(x) for x in groups))
        if g and not (0 <= g[0] and g[-1] < self.remnant.n_groups):
            raise ValueError("group ids outside [0, " + str(self.remnant.n_groups)
                             + "): " + str(g[:5]))
        self.groups = g

    @property
    def active_fraction(self) -> float:
        return len(self.groups) / self.remnant.n_groups

    def forward(self, x: Tensor) -> Tensor:
        if not self.groups:
            # Exact deletion of this sub-block. No weight is read.
            return torch.zeros_like(x)
        return self.remnant.forward_groups(x, self.groups)

    def extra_repr(self) -> str:
        return (str(len(self.groups)) + "/" + str(self.remnant.n_groups)
                + " groups active")


# --------------------------------------------------------------------------
@dataclass
class RetirementPlan:
    """Which layers are retired, at what precision, with which donors.

    ``donor_map`` lets a retired layer be filled with **another** layer's
    weights, which is the layer-identity control (arm G). It is part of the plan
    rather than a separate code path so that the wrong-layer arm travels exactly
    the same machinery as the correct one -- a difference between arms must be a
    difference of configuration, never of implementation.
    """

    layers: tuple[int, ...]
    spec: QuantSpec = field(default_factory=lambda: QuantSpec(4, 128, False))
    n_groups: int = 24
    donor_map: dict[int, int] | None = None      # retired layer -> donor layer

    def donor_for(self, layer: int) -> int:
        return (self.donor_map or {}).get(layer, layer)

    @property
    def is_wrong_layer(self) -> bool:
        return any(self.donor_for(l) != l for l in self.layers)

    def describe(self) -> str:
        core = "L" + "/".join(str(l) for l in self.layers) + " " + self.spec.name()
        if self.is_wrong_layer:
            core += "  donors " + ",".join(
                str(l) + "<-" + str(self.donor_for(l)) for l in self.layers)
        return core


@dataclass
class InstalledSGQR:
    """Handle for a set of retired layers, so a run can report and undo itself."""

    model: nn.Module
    plan: RetirementPlan
    modules: dict[int, SGQRMlp]
    originals: dict[int, nn.Module]
    donor_freed: bool

    # -- budget ------------------------------------------------------------
    def set_allocation(self, alloc: dict[int, list[int]]) -> None:
        """Set the active group set of every retired layer at once."""
        for layer, groups in alloc.items():
            self.modules[layer].set_groups(groups)

    def active_groups(self) -> int:
        return sum(len(m.groups) for m in self.modules.values())

    def total_groups(self) -> int:
        return sum(m.remnant.n_groups for m in self.modules.values())

    @property
    def active_fraction(self) -> float:
        t = self.total_groups()
        return self.active_groups() / t if t else 0.0

    # -- accounting --------------------------------------------------------
    def packed_bytes(self) -> int:
        return sum(m.remnant.packed_bytes()["total"] for m in self.modules.values())

    def dense_bytes(self) -> int:
        return sum(m.remnant.dense_bytes() for m in self.modules.values())

    def active_flops_per_token(self) -> int:
        """MACs actually executed per token across all retired layers.

        Executed work, not mathematical sparsity: an inactive group's rows are
        never dequantized and never enter a matmul, which CSGQR-v1 verified by
        instrumenting ``PackedMatrix.dequantize`` and counting rows touched.
        """
        return sum(m.remnant.active_flops(m.groups, 1) for m in self.modules.values())

    def dense_mlp_flops_per_token(self) -> int:
        """What the same layers would cost undeleted, for the saving ratio."""
        return sum(m.remnant.active_flops(list(range(m.remnant.n_groups)), 1)
                   for m in self.modules.values())

    def report(self) -> dict[str, object]:
        return {
            "layers": list(self.plan.layers),
            "donor_map": {str(l): self.plan.donor_for(l) for l in self.plan.layers},
            "wrong_layer": self.plan.is_wrong_layer,
            "spec": self.plan.spec.name(),
            "bits": self.plan.spec.bits,
            "n_groups_per_layer": self.plan.n_groups,
            "active_groups": self.active_groups(),
            "total_groups": self.total_groups(),
            "active_fraction": self.active_fraction,
            "per_layer_active": {str(l): len(m.groups)
                                 for l, m in self.modules.items()},
            "packed_bytes": self.packed_bytes(),
            "dense_bytes": self.dense_bytes(),
            "compression_vs_dense": (self.dense_bytes()
                                     / max(self.packed_bytes(), 1)),
            "active_flops_per_token": self.active_flops_per_token(),
            "dense_mlp_flops_per_token": self.dense_mlp_flops_per_token(),
            "donor_freed": self.donor_freed,
        }

    def remove(self) -> None:
        if self.donor_freed:
            raise RuntimeError(
                "cannot restore: the full-precision donors were freed at install "
                "time. Reload the model to get an unmodified copy.")
        for layer, original in self.originals.items():
            self.model.model.layers[layer].mlp = original


# --------------------------------------------------------------------------
def install_sgqr(model: nn.Module, plan: RetirementPlan, *,
                 allocation: dict[int, list[int]] | None = None,
                 keep_donor_resident: bool = False) -> InstalledSGQR:
    """Retire every layer in ``plan``, replacing each MLP with a remnant.

    ``allocation`` gives the initially active groups per layer; the default is
    every group (arm C, the full remnant). Pass ``{l: [] for l in layers}`` for
    exact deletion (arm B) -- which travels this same code path deliberately, so
    the deletion baseline cannot differ from the routed arms by anything except
    the group set.

    ``keep_donor_resident=False`` drops the original bf16 projections after
    quantizing. That is the honest configuration for a memory measurement:
    leaving an unreferenced dense copy alive would make every VRAM number
    meaningless while every quality number stayed identical.
    """
    n_layers = len(model.model.layers)
    for l in plan.layers:
        if not 0 <= l < n_layers:
            raise ValueError("layer " + str(l) + " outside [0, " + str(n_layers) + ")")
        d = plan.donor_for(l)
        if not 0 <= d < n_layers:
            raise ValueError("donor layer " + str(d) + " out of range")

    modules: dict[int, SGQRMlp] = {}
    originals: dict[int, nn.Module] = {}

    # Build every remnant BEFORE freeing any donor. A wrong-layer plan may use a
    # donor that is itself retired, and freeing in-place would then quantize an
    # already-hollowed module.
    for l in plan.layers:
        target = model.model.layers[l]
        source = model.model.layers[plan.donor_for(l)].mlp
        dtype = target.mlp.gate_proj.weight.dtype
        device = target.mlp.gate_proj.weight.device
        rem = QuantizedMlpRemnant(
            source, RemnantConfig(layer=l, n_groups=plan.n_groups,
                                  spec=plan.spec, compute_dtype=dtype)).to(device)
        groups = (list(range(plan.n_groups)) if allocation is None
                  else list(allocation.get(l, [])))
        modules[l] = SGQRMlp(rem, groups)
        originals[l] = target.mlp

    for l in plan.layers:
        model.model.layers[l].mlp = modules[l]

    if not keep_donor_resident:
        for l, original in originals.items():
            for name in ("gate_proj", "up_proj", "down_proj"):
                if hasattr(original, name):
                    delattr(original, name)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return InstalledSGQR(model=model, plan=plan, modules=modules,
                         originals=originals, donor_freed=not keep_donor_resident)


# --------------------------------------------------------------------------
# Budget allocation
# --------------------------------------------------------------------------
def allocate_global(scores: dict[int, list[float]], fraction: float,
                    *, n_groups: int) -> dict[int, list[int]]:
    """One top-k over ALL groups of ALL retired layers, ranked by ``scores``.

    This is what lets significance decide that one retired layer deserves more of
    the budget than another. Ties resolve deterministically by (layer, group),
    because blockwise quantization produces exact ties more often than continuous
    data would and an arbitrary tie-break makes an arm irreproducible for exactly
    the choices that are arbitrary.
    """
    flat: list[tuple[float, int, int]] = []
    for layer, vec in scores.items():
        if len(vec) != n_groups:
            raise ValueError("layer " + str(layer) + " has " + str(len(vec))
                             + " scores; expected " + str(n_groups))
        for g, s in enumerate(vec):
            flat.append((float(s), layer, g))
    total = len(flat)
    k = _budget_k(fraction, total)
    # -score for descending; (layer, group) ascending breaks ties.
    flat.sort(key=lambda t: (-t[0], t[1], t[2]))
    out: dict[int, list[int]] = {l: [] for l in scores}
    for _, layer, g in flat[:k]:
        out[layer].append(g)
    return {l: sorted(v) for l, v in out.items()}


def allocate_uniform(scores: dict[int, list[float]], fraction: float,
                     *, n_groups: int) -> dict[int, list[int]]:
    """The same rho inside every layer. Control for the cross-layer allocation."""
    k = _budget_k(fraction, n_groups)
    out: dict[int, list[int]] = {}
    for layer, vec in scores.items():
        order = sorted(range(len(vec)), key=lambda g: (-float(vec[g]), g))
        out[layer] = sorted(order[:k])
    return out


def allocate_random(layers: tuple[int, ...], fraction: float, *, n_groups: int,
                    seed: int) -> dict[int, list[int]]:
    """Matched-compute random control: the same total k, chosen at random.

    Drawn over the flattened (layer, group) space rather than per layer, so it is
    matched to ``allocate_global`` in both total budget *and* in being free to
    distribute unevenly across layers. A per-layer-uniform random control would
    be a weaker comparison and would confound "which groups" with "which layers".
    """
    total = len(layers) * n_groups
    k = _budget_k(fraction, total)
    gen = torch.Generator().manual_seed(seed)
    pick = torch.randperm(total, generator=gen)[:k].tolist()
    out: dict[int, list[int]] = {l: [] for l in layers}
    for idx in pick:
        out[layers[idx // n_groups]].append(idx % n_groups)
    return {l: sorted(v) for l, v in out.items()}


def _budget_k(fraction: float, total: int) -> int:
    """Groups to activate for a fraction of ``total``.

    The ``- 1e-9`` before the ceiling is the fix from CSGQR-v1 S7: callers express
    an exact k as ``k / total``, and that round-trip overshoots for many k
    (``5/24 * 24 == 5.000000000000001``), so a bare ceil silently activates k+1
    while the run is labelled k.
    """
    import math
    if fraction <= 0.0:
        return 0
    if fraction >= 1.0:
        return total
    return min(total, max(1, int(math.ceil(fraction * total - 1e-9))))


def magnitude_scores(inst: InstalledSGQR) -> dict[int, list[float]]:
    """Frobenius norm of each group's dequantized ``down`` rows.

    The cheap static proxy that interventional significance has to beat. In
    CSGQR-v1 at one layer it was *worse than random*, which is the result this
    control exists to re-test at four-layer scale.
    """
    out: dict[int, list[float]] = {}
    for layer, mod in inst.modules.items():
        rem = mod.remnant
        vals = []
        for g in range(rem.n_groups):
            lo, hi = rem.bounds[g]
            vals.append(rem.down_q.dequantize(lo, hi).to(torch.float32).norm().item())
        out[layer] = vals
    return out
