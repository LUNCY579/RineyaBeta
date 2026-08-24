#!/usr/bin/env python3
"""The frozen quantized MLP remnant, decomposed into additive structured groups.

The additivity that CSGQR rests on
----------------------------------
Qwen3's MLP, with ``bias=False`` on all three projections, is::

    F(x) = W_down @ ( silu(W_gate x) * (W_up x) )

Index the intermediate dimension by ``i``. The elementwise product is per-``i``,
and ``W_down`` is linear over ``i``, so for any partition ``{I_g}`` of the
intermediate axis::

    F(x) = sum_g  W_down[:, I_g] @ ( silu(W_gate[I_g, :] x) * (W_up[I_g, :] x) )
         = sum_g  F_g(x)

*exactly*, not approximately -- the only discrepancy is floating-point summation
order. So executing a subset of groups is mathematically identical to selecting
that subset of intermediate neurons, and there is no cross-group interaction
term to model. ``verify_csgqr.py`` tests this rather than assuming it, because
the whole routing story is void if it fails.

The absence of a bias is what makes this exact. A bias on ``down_proj`` would be
counted once per active group instead of once in total; the code asserts
``bias is None`` rather than silently producing that error.

What is actually skipped
------------------------
:meth:`QuantizedMlpRemnant.forward_groups` slices the packed bytes of the
inactive groups away *before* dequantizing and before any matmul (see
``quant.PackedMatrix.dequantize``, which takes a row range). Nothing is computed
and then masked. At rho = 0.25 the three matmuls really are a quarter of the
size. Whether that translates into wall-clock savings on a given card is a
separate, measured question -- see ``bench_gpu.py`` -- and the answer on
Ada-class hardware with plain PyTorch is not automatically yes, because the
dequantization is itself work that the dense bf16 path does not do.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from quant import PackedMatrix, QuantSpec


@dataclass
class RemnantConfig:
    """Shape and precision of a remnant.

    ``n_groups`` partitions the intermediate axis. Group size need not divide
    the intermediate dimension evenly -- the final group absorbs the remainder
    and :func:`group_bounds` reports its true extent -- but the common case on
    this checkpoint does divide evenly (3072 = 24 x 128 = 48 x 64).
    """

    layer: int
    n_groups: int = 24
    spec: QuantSpec = field(default_factory=lambda: QuantSpec(4, 128, False))
    compute_dtype: torch.dtype = torch.bfloat16

    def name(self) -> str:
        return ("L" + str(self.layer) + "-" + self.spec.name()
                + "-g" + str(self.n_groups))


def group_bounds(intermediate: int, n_groups: int) -> list[tuple[int, int]]:
    """Half-open row ranges for each group, remainder folded into the last.

    Returned ranges are contiguous, ascending, and cover ``[0, intermediate)``
    exactly once -- the partition property the additivity argument needs.
    """
    if not 1 <= n_groups <= intermediate:
        raise ValueError("n_groups must be in [1, " + str(intermediate) + "]; got "
                         + str(n_groups))
    size = intermediate // n_groups
    bounds = [(g * size, (g + 1) * size) for g in range(n_groups)]
    lo, _ = bounds[-1]
    bounds[-1] = (lo, intermediate)
    return bounds


class QuantizedMlpRemnant(nn.Module):
    """A frozen, packed, group-addressable copy of one Qwen3 MLP.

    All three projections are stored as ``[intermediate, hidden]`` so that a
    group is a contiguous row slice of each (``quant.py`` explains why
    ``down_proj`` is transposed). ``alpha`` is a calibrated output gain, held as
    a per-group vector so a scalar gain and a per-group gain travel the same
    code path.
    """

    def __init__(self, mlp: nn.Module, config: RemnantConfig) -> None:
        super().__init__()
        for name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(mlp, name)
            if getattr(proj, "bias", None) is not None:
                raise ValueError(
                    name + " has a bias. Group decomposition is only additive for "
                    "bias-free projections: a down_proj bias would be added once "
                    "per active group instead of once overall. Refusing rather "
                    "than silently producing that error.")

        self.config = config
        self.intermediate, self.hidden = mlp.gate_proj.weight.shape
        self.bounds = group_bounds(self.intermediate, config.n_groups)
        self.n_groups = config.n_groups

        dt = config.compute_dtype
        self.gate_q = PackedMatrix(mlp.gate_proj.weight, config.spec, compute_dtype=dt)
        self.up_q = PackedMatrix(mlp.up_proj.weight, config.spec, compute_dtype=dt)
        # Transposed: [intermediate, hidden], so a group is a row slice here too.
        self.down_q = PackedMatrix(mlp.down_proj.weight.t().contiguous(),
                                   config.spec, compute_dtype=dt)

        self.register_buffer("alpha", torch.ones(config.n_groups, dtype=torch.float32))

    # -- accounting --------------------------------------------------------
    def packed_bytes(self) -> dict[str, int]:
        """Resident bytes, itemized by matrix. Not a parameter count."""
        parts = {"gate": self.gate_q.packed_bytes(),
                 "up": self.up_q.packed_bytes(),
                 "down": self.down_q.packed_bytes()}
        total = sum(p["total"] for p in parts.values())
        alpha = self.alpha.numel() * self.alpha.element_size()
        return {**{k: v["total"] for k, v in parts.items()},
                "alpha": alpha, "total": total + alpha,
                "scales": sum(p["scales"] for p in parts.values()),
                "zeros": sum(p["zeros"] for p in parts.values())}

    def dense_bytes(self) -> int:
        """What the same three matrices cost unquantized at compute dtype."""
        return (self.gate_q.dense_bytes() + self.up_q.dense_bytes()
                + self.down_q.dense_bytes())

    def group_flops(self, tokens: int = 1, group: int | None = None) -> int:
        """Multiply-accumulates for one group, per token.

        Three matmuls of ``[hidden] x [group_size, hidden]``, counted as
        2*m*n MACs the usual way. Reported per group so ``rho`` maps onto a
        FLOP number without re-deriving it at each call site.

        ``group=None`` reports the nominal group size. Pass a group id for the
        true size, which differs for the last group when ``n_groups`` does not
        divide the intermediate dimension -- the remainder is folded there, so
        quoting the nominal size would under-count exactly one group. It divides
        evenly on this checkpoint (3072 = 24 x 128), so the two agree here; the
        argument exists so that a future uneven partition cannot silently
        under-report its own cost.
        """
        if group is None:
            size = self.intermediate // self.n_groups
        else:
            lo, hi = self.bounds[group]
            size = hi - lo
        return 2 * tokens * size * self.hidden * 3

    def active_flops(self, groups: list[int], tokens: int = 1) -> int:
        """MACs for executing exactly ``groups``, per ``tokens`` tokens.

        Sums the true per-group cost, so an uneven partition is accounted
        correctly. This is *executed* work, not mathematical sparsity: the rows
        of an inactive group are never dequantized and never enter a matmul.
        """
        return sum(self.group_flops(tokens, g) for g in sorted(set(groups)))

    # -- execution ---------------------------------------------------------
    @staticmethod
    def _runs(bounds: list[tuple[int, int]],
              groups: list[int]) -> list[tuple[int, int]]:
        """Merge the selected groups' row ranges into maximal contiguous runs.

        ``groups`` must be sorted. Adjacent group ids give adjacent row ranges,
        so a selection like ``[3, 4, 5, 9]`` is two runs, not four. This is why
        the layout in ``quant.py`` puts all three projections in
        ``[intermediate, hidden]``: a run is then one slice of each.
        """
        runs: list[tuple[int, int]] = []
        for g in groups:
            lo, hi = bounds[g]
            if runs and runs[-1][1] == lo:
                runs[-1] = (runs[-1][0], hi)
            else:
                runs.append((lo, hi))
        return runs

    def _gather(self, groups: list[int]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Dequantize only the requested groups. Returns (Wg, Wu, Wd, alpha_vec).

        The row slice happens inside ``PackedMatrix.dequantize``, so the packed
        bytes of an inactive group are never read and its rows are never
        materialized.

        Contiguous groups are coalesced into one dequantization call per run
        rather than one per group. This is purely a launch-overhead win and
        changes no arithmetic -- the concatenation order is identical either way,
        because runs are built from sorted groups and each run is a contiguous
        ascending row range. It matters because the reference backend was issuing
        ``3 * len(groups)`` dequantization calls per distinct group set per token
        bucket, and at ``rho = 1`` that is 72 launches to reconstruct three
        matrices that are one slice each. ``verify_csgqr.py`` asserts the
        coalesced and per-group gathers are bit-identical.
        """
        runs = self._runs(self.bounds, groups)
        gate_parts, up_parts, down_parts = [], [], []
        for lo, hi in runs:
            gate_parts.append(self.gate_q.dequantize(lo, hi))
            up_parts.append(self.up_q.dequantize(lo, hi))
            down_parts.append(self.down_q.dequantize(lo, hi))
        alpha_parts = [self.alpha[g].expand(hi - lo)
                       for g in groups for (lo, hi) in (self.bounds[g],)]
        cat = (lambda parts: parts[0] if len(parts) == 1
               else torch.cat(parts, dim=0))
        return (cat(gate_parts), cat(up_parts), cat(down_parts),
                cat(alpha_parts))

    def forward_groups(self, x: Tensor, groups: list[int]) -> Tensor:
        """Compute ``sum_{g in groups} alpha_g * F_g(x)``.

        ``x`` is the *normalized* MLP input, matching what Qwen3 feeds its MLP.
        An empty group list returns exact zeros without touching any weight.
        """
        if not groups:
            return torch.zeros_like(x)
        w_gate, w_up, w_down, alpha = self._gather(sorted(set(groups)))
        act = torch.nn.functional.silu(x @ w_gate.t()) * (x @ w_up.t())
        act = act * alpha.to(act.dtype)
        return act @ w_down

    def forward(self, x: Tensor) -> Tensor:
        """The full remnant: every group active."""
        return self.forward_groups(x, list(range(self.n_groups)))

    def group_outputs(self, x: Tensor) -> Tensor:
        """Per-group contributions, shape ``[n_groups, *x.shape[:-1], hidden]``.

        Used by the utility-labelling pass, which needs each ``F_g(x)``
        separately. Deliberately not the deployment path -- it computes every
        group -- and callers that want a routed forward must use
        :meth:`forward_groups`.
        """
        outs = [self.forward_groups(x, [g]) for g in range(self.n_groups)]
        return torch.stack(outs, dim=0)

    def extra_repr(self) -> str:
        pb = self.packed_bytes()["total"]
        return (self.config.name() + ", " + str(self.intermediate) + "x"
                + str(self.hidden) + ", " + str(self.n_groups) + " groups, "
                + format(pb, ",") + " B packed vs " + format(self.dense_bytes(), ",")
                + " B dense")


@torch.no_grad()
def reference_group_output(mlp: nn.Module, x: Tensor, lo: int, hi: int) -> Tensor:
    """``F_g(x)`` computed from the *unquantized* donor, for rows ``[lo, hi)``.

    The full-precision counterpart of one remnant group. Needed because the
    per-group gain in the specification is fitted against ``F_{j,g}``, the
    donor's own group output -- not against the whole MLP. Using the whole MLP as
    each group's target is a different and wrong estimator; see
    :func:`calibrate_alpha`.
    """
    w_gate = mlp.gate_proj.weight[lo:hi]
    w_up = mlp.up_proj.weight[lo:hi]
    w_down = mlp.down_proj.weight[:, lo:hi]
    act = torch.nn.functional.silu(x @ w_gate.t()) * (x @ w_up.t())
    return act @ w_down.t()


@torch.no_grad()
def calibrate_alpha(remnant: QuantizedMlpRemnant, reference_mlp: nn.Module,
                    inputs: Tensor, *, per_group: bool = False,
                    subset: list[int] | None = None) -> Tensor:
    """Least-squares output gain for the quantized remnant.

    Scalar form, as specified::

        alpha = E<F^Q(h), F(h)> / ( E||F^Q(h)||^2 + eps )

    which is the gain minimising ``||F(h) - alpha F^Q(h)||^2``. It corrects the
    systematic shrinkage that blockwise quantization introduces; it is *not* a
    trained parameter, and it is fitted on calibration inputs only.

    ``per_group=True`` solves the same 1-D problem per group, each against **its
    own** donor group output ``F_{j,g}``::

        alpha_g = E<F^Q_{j,g}(h), F_{j,g}(h)> / ( E||F^Q_{j,g}(h)||^2 + eps )

    That is what the specification asks for. An earlier revision of this function
    fitted every group against the *whole* MLP output ``F(h)``; because
    ``<F_g, F> = ||F_g||^2 + <F_g, sum_{h != g} F_h>``, that estimator is
    inflated by the cross-group inner product and does not converge to 1 even for
    a lossless remnant. It is a different estimator, not a looser one, and it
    biased every per-group gain upward. Regression-tested in
    ``verify_csgqr.py``: a bit-exact (bits=16) remnant must return alpha = 1 for
    every group, which the old form failed.

    ``subset`` fits a *scalar* gain for the specific active group set, against
    the donor's output restricted to the same rows. This is the gain a
    partial-execution arm actually needs: at ``rho < 1`` the remnant reproduces
    only part of ``F``, and the right question is not "how do I rescale a partial
    sum to look like the whole MLP" -- it is "given that I am spending k groups,
    what scalar makes those k groups' output closest to what those same k donor
    channels would have produced". Fitting a partial sum against the *whole*
    donor output would inflate the gain by roughly ``1/rho`` and is not offered.

    Numerics. The design writes the denominator as ``E||F^Q(h)||^2 + eps``. An
    *absolute* eps is wrong here and measurably so: MLP output magnitudes depend
    on the checkpoint's weight scale, and on a small-activation layer a fixed
    ``1e-8`` was 13% of the denominator, pulling a true alpha of 0.998 down to
    0.880 and making the calibrated remnant *worse* than an uncalibrated one.
    So the degenerate case is handled as the zero-check it actually is, and the
    sums accumulate in float64 because they run over hundreds of thousands of
    tokens where a float32 reduction visibly drifts.

    The remnant's current ``alpha`` is neutralised for the duration of the fit,
    so calling this twice is idempotent rather than compounding.
    """
    flat = inputs.reshape(-1, inputs.shape[-1])

    def gain(approx: Tensor, target: Tensor) -> float:
        wide = approx.to(torch.float64)
        den = wide.pow(2).sum().item()
        if den <= 0.0:
            # The remnant contributes exactly nothing on this data, so every
            # gain is equally optimal; 1.0 keeps the arm interpretable.
            return 1.0
        return (wide * target.to(torch.float64)).sum().item() / den

    saved = remnant.alpha.clone()
    remnant.alpha.fill_(1.0)
    try:
        if subset is not None:
            groups = sorted(set(subset))
            if not groups:
                return saved
            ref = sum(reference_group_output(reference_mlp, flat,
                                             *remnant.bounds[g]) for g in groups)
            value = gain(remnant.forward_groups(flat, groups), ref)
            return torch.full((remnant.n_groups,), value, dtype=torch.float32)

        if not per_group:
            whole = reference_mlp(flat)
            value = gain(remnant.forward_groups(flat, list(range(remnant.n_groups))),
                         whole)
            return torch.full((remnant.n_groups,), value, dtype=torch.float32)

        gains = torch.ones(remnant.n_groups, dtype=torch.float32)
        for g in range(remnant.n_groups):
            lo, hi = remnant.bounds[g]
            gains[g] = gain(remnant.forward_groups(flat, [g]),
                            reference_group_output(reference_mlp, flat, lo, hi))
        return gains
    finally:
        remnant.alpha.copy_(saved)
