#!/usr/bin/env python3
"""Verification suite for the CSGQR implementation. CPU-only, deterministic.

Every check here is a property the method's *claims* depend on, not a smoke
test. Failures are reported with the measured number so the failure can be
diagnosed rather than merely noticed, and the suite exits non-zero so it can
gate a commit.

Run:
    python experiments/csgqr/src/verify_csgqr.py
    python experiments/csgqr/src/verify_csgqr.py --json results/verify.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from budget import ModelShape, remnant_storage                    # noqa: E402
from confidence import (AlignmentLens, ConfidenceHead,             # noqa: E402
                        expected_calibration_error, features_from_logits,
                        fit_temperature, js_divergence, stability)
from quant import PackedMatrix, QuantSpec, _pack, _unpack          # noqa: E402
from remnant import (QuantizedMlpRemnant, RemnantConfig,           # noqa: E402
                     calibrate_alpha, group_bounds)
from router import RoutingPolicy, RoutingTrace                     # noqa: E402
from significance import (GroupSignificance, kl_divergence,        # noqa: E402
                          magnitude_ranking, marginal_utility,
                          rank_correlation, reversed_ranking, shuffled_ranking)

RESULTS: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append({"name": name, "ok": bool(ok), "detail": detail})
    mark = "PASS" if ok else "FAIL"
    print("  [" + mark + "] " + name + ("   " + detail if detail else ""))
    return bool(ok)


def section(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def _raises(fn, exc: type[BaseException]) -> bool:
    """True when ``fn()`` raises ``exc``. Keeps refusal checks to one line."""
    try:
        fn()
    except exc:
        return True
    except Exception:  # noqa: BLE001 - the wrong exception type is a failure
        return False
    return False


def toy_mlp(hidden: int = 256, inter: int = 768, seed: int = 0,
            scale: float = 1.0) -> nn.Module:
    """A bias-free gated MLP with the same algebra as Qwen3MLP."""
    torch.manual_seed(seed)

    class Gated(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(hidden, inter, bias=False)
            self.up_proj = nn.Linear(hidden, inter, bias=False)
            self.down_proj = nn.Linear(inter, hidden, bias=False)
            self.act_fn = nn.SiLU()

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    m = Gated().to(torch.float32).eval()
    for p in m.parameters():
        p.data.mul_(scale)
    return m


# ==========================================================================
# A. Quantization correctness
# ==========================================================================
def verify_quantization() -> None:
    section("A. Quantization: packing, signs, scales, zeros, non-divisible shapes")
    torch.manual_seed(0)

    ok = True
    for cols in (1024, 1023, 129, 7, 2, 1):
        codes = torch.randint(0, 16, (5, cols), dtype=torch.int64)
        spec = QuantSpec(4, 128, False)
        packed = _pack(codes, spec)
        back = _unpack(packed, cols, spec).to(torch.int64)
        exact = torch.equal(back, codes)
        expected_bytes = (cols + 1) // 2
        ok &= exact and packed.shape[1] == expected_bytes
    check("4-bit pack/unpack is a bijection at even and odd widths", ok)

    codes8 = torch.randint(0, 256, (4, 100), dtype=torch.int64)
    s8 = QuantSpec(8, 128, False)
    check("8-bit pack/unpack is a bijection",
          torch.equal(_unpack(_pack(codes8, s8), 100, s8).to(torch.int64), codes8))

    # Exact-zero preservation and dead blocks.
    w = torch.randn(6, 256) * 0.02
    w[1, :128] = 0.0
    w[2, :] = 0.0
    w[3, 17] = 0.0
    dead_ok = zero_ok = True
    for spec in (QuantSpec(4, 128, True), QuantSpec(4, 128, False),
                 QuantSpec(8, 128, True), QuantSpec(8, 128, False),
                 QuantSpec(4, 64, False)):
        r = PackedMatrix(w, spec, compute_dtype=torch.float32).dequantize()
        dead_ok &= r[1, :128].abs().max().item() == 0.0
        dead_ok &= r[2].abs().max().item() == 0.0
        if not spec.symmetric:
            zero_ok &= r[3, 17].item() == 0.0
    check("all-zero blocks and rows dequantize to exact zero (no div-by-zero)", dead_ok)
    check("asymmetric form preserves an isolated exact zero", zero_ok)

    # Non-divisible input dim.
    wn = torch.randn(32, 1000) * 0.02
    pm = PackedMatrix(wn, QuantSpec(4, 128, False), compute_dtype=torch.float32)
    err = (pm.dequantize() - wn).norm().item() / wn.norm().item()
    check("non-divisible input dim (1000 = 7x128 + 104) quantizes",
          pm.n_blocks == 8 and pm.scales.shape == (32, 8) and err < 0.2,
          "n_blocks=" + str(pm.n_blocks) + " rel_err=" + format(err, ".4f"))

    # Row-slice equals the corresponding rows of the full dequantization.
    big = torch.randn(768, 256) * 0.02
    pm = PackedMatrix(big, QuantSpec(4, 128, False), compute_dtype=torch.float32)
    full = pm.dequantize()
    slice_ok = all(torch.equal(pm.dequantize(lo, hi), full[lo:hi])
                   for lo, hi in ((0, 32), (32, 64), (700, 768), (0, 768), (5, 5)))
    check("row-slice dequantization equals the full dequantization on that slice",
          slice_ok)

    # Monotone fidelity in bitwidth, and the ordering asym <= sym at 4 bits.
    errs = {}
    for spec in (QuantSpec(8, 128, False), QuantSpec(4, 128, False),
                 QuantSpec(4, 128, True), QuantSpec(2, 128, False)):
        r = PackedMatrix(big, spec, compute_dtype=torch.float32).dequantize()
        errs[spec.name()] = ((r - big).norm() / big.norm()).item()
    check("fidelity improves monotonically with bitwidth",
          errs["int8-b128-asym"] < errs["int4-b128-asym"] < errs["int2-b128-asym"],
          " ".join(k + "=" + format(v, ".4f") for k, v in errs.items()))

    # Smaller blocks must not be worse.
    e_b32 = ((PackedMatrix(big, QuantSpec(4, 32, False), compute_dtype=torch.float32)
              .dequantize() - big).norm() / big.norm()).item()
    check("smaller quantization blocks reduce error",
          e_b32 <= errs["int4-b128-asym"],
          "b32=" + format(e_b32, ".4f") + " vs b128="
          + format(errs["int4-b128-asym"], ".4f"))

    # No full-precision residue, no parameters, no optimizer-visible state.
    pm4 = PackedMatrix(big, QuantSpec(4, 128, False))
    floats = [n for n, b in pm4.named_buffers()
              if b.dtype in (torch.float32, torch.float64, torch.bfloat16)]
    params = list(dict(pm4.named_parameters()).keys())
    check("quantized matrix retains no fp32/bf16 weight buffer and no parameters",
          not floats and not params,
          "float_buffers=" + str(floats) + " params=" + str(params))
    check("no gradient can be created for packed storage",
          not any(b.requires_grad for _, b in pm4.named_buffers()))

    # Byte accounting matches the analytic model.
    pb = pm4.packed_bytes()
    expect_w = 768 * (256 // 2)
    expect_s = 768 * 2 * 2
    expect_z = 768 * 2
    check("packed byte accounting is exact",
          pb["weight"] == expect_w and pb["scales"] == expect_s
          and pb["zeros"] == expect_z,
          str(pb))


# ==========================================================================
# B. Structured MLP decomposition
# ==========================================================================
def verify_decomposition() -> None:
    section("B. Group decomposition: exact additivity and true neuron selection")
    mlp = toy_mlp()
    torch.manual_seed(1)
    x = torch.randn(16, 256) * 0.7

    for inter, ng in ((768, 24), (768, 48), (768, 7), (768, 1), (768, 768), (1000, 7)):
        b = group_bounds(inter, ng)
        cover = [i for lo, hi in b for i in range(lo, hi)]
        if not (sorted(cover) == list(range(inter)) and len(cover) == inter
                and len(b) == ng):
            check("group_bounds partitions exactly (inter=" + str(inter)
                  + ", n=" + str(ng) + ")", False)
            break
    else:
        check("group_bounds is an exact partition at every tested shape, "
              "including non-divisible", True)

    r16 = QuantizedMlpRemnant(mlp, RemnantConfig(0, 24, QuantSpec(16, 0, True),
                                                 torch.float32))
    ref = mlp(x)
    check("fp passthrough remnant reproduces the source MLP bit-exactly",
          torch.equal(r16.forward(x), ref))

    # Additivity is claimed to be *exact* algebra, so it is tested where the
    # arithmetic can show that: float64. Summing 24 group outputs in float32
    # necessarily accumulates ~1e-8 on order-1 values, which is float32 epsilon
    # and not a property of the decomposition -- so float32 is checked on a
    # relative tolerance and float64 carries the exactness claim.
    mlp64 = toy_mlp().to(torch.float64)
    x64 = x.to(torch.float64)
    r64 = QuantizedMlpRemnant(mlp64, RemnantConfig(0, 24, QuantSpec(16, 0, True),
                                                   torch.float64))
    ref64 = mlp64(x64)
    rel64 = ((r64.group_outputs(x64).sum(dim=0) - ref64).norm()
             / ref64.norm()).item()
    check("sum of group outputs equals the full MLP in float64 (exact algebra)",
          rel64 < 1e-14, "rel_err=" + format(rel64, ".3e"))

    summed = r16.group_outputs(x).sum(dim=0)
    rel32 = ((summed - ref).norm() / ref.norm()).item()
    check("additivity holds in float32 to within float32 epsilon",
          rel32 < 1e-6, "rel_err=" + format(rel32, ".3e"))

    # Partial groups must equal explicit intermediate-neuron selection.
    bounds = group_bounds(768, 24)

    def explicit(sel: list[int], w_mlp, xx) -> torch.Tensor:
        idx = torch.tensor([i for g in sel for i in range(*bounds[g])])
        g = torch.nn.functional.silu(xx @ w_mlp.gate_proj.weight[idx].t())
        u = xx @ w_mlp.up_proj.weight[idx].t()
        return (g * u) @ w_mlp.down_proj.weight[:, idx].t()

    subsets = ([0], [23], [3, 7], [0, 1, 2], list(range(0, 24, 2)),
               list(range(24)), [11, 2, 19])
    worst64 = max(((r64.forward_groups(x64, s) - explicit(s, mlp64, x64)).norm()
                   / explicit(s, mlp64, x64).norm()).item() for s in subsets)
    check("partial group execution == explicit intermediate-neuron selection "
          "(float64)", worst64 < 1e-14, "rel_err=" + format(worst64, ".3e"))

    check("empty group set returns exact zeros without touching any weight",
          torch.equal(r16.forward_groups(x, []), torch.zeros_like(x)))
    check("duplicate group ids are deduplicated, not double-counted",
          torch.equal(r16.forward_groups(x, [3, 3, 3]),
                      r16.forward_groups(x, [3])))
    check("group order does not change the result",
          torch.allclose(r16.forward_groups(x, [1, 5, 9]),
                         r16.forward_groups(x, [9, 1, 5]), atol=0, rtol=0))

    # Quantization must not disturb additivity: the groups partition the same
    # packed rows, so the property is independent of the values in them.
    r4 = QuantizedMlpRemnant(mlp64, RemnantConfig(0, 24, QuantSpec(4, 128, False),
                                                  torch.float64))
    full4 = r4.forward(x64)
    rel4 = ((r4.group_outputs(x64).sum(dim=0) - full4).norm() / full4.norm()).item()
    check("additivity also holds for the quantized remnant (float64)",
          rel4 < 1e-14, "rel_err=" + format(rel4, ".3e"))

    # A bias would break additivity; the constructor must refuse.
    bad = toy_mlp()
    bad.down_proj = nn.Linear(768, 256, bias=True)
    try:
        QuantizedMlpRemnant(bad, RemnantConfig(0, 24))
        check("a biased down_proj is refused (would break additivity)", False)
    except ValueError:
        check("a biased down_proj is refused (would break additivity)", True)

    # -- coalesced dequantization ------------------------------------------
    # Contiguous groups are merged into one dequantization call per run. That is
    # a launch-count optimisation and must be bit-identical, not merely close:
    # if it were only close, every quantized arm's quality would depend on how
    # the router happened to group its selection.
    rc = QuantizedMlpRemnant(mlp64, RemnantConfig(0, 24, QuantSpec(4, 128, False),
                                                  torch.float64))

    def per_group_gather(rem, groups):
        gs, us, ds, als = [], [], [], []
        for g in sorted(set(groups)):
            lo, hi = rem.bounds[g]
            gs.append(rem.gate_q.dequantize(lo, hi))
            us.append(rem.up_q.dequantize(lo, hi))
            ds.append(rem.down_q.dequantize(lo, hi))
            als.append(rem.alpha[g].expand(hi - lo))
        return (torch.cat(gs), torch.cat(us), torch.cat(ds), torch.cat(als))

    sets = ([0], [0, 1, 2, 3], [3, 4, 5, 9], [5, 7, 9, 11], [0, 23],
            list(range(24)))
    identical = True
    for sel in sets:
        a = rc._gather(sorted(set(sel)))
        b = per_group_gather(rc, sel)
        identical &= all(torch.equal(p, q) for p, q in zip(a, b))
    check("coalesced run-merged gather is bit-identical to a per-group gather",
          identical, str(len(sets)) + " group sets")

    runs_full = rc._runs(rc.bounds, list(range(24)))
    runs_split = rc._runs(rc.bounds, [5, 7, 9, 11])
    check("contiguous groups merge into one run; non-contiguous do not",
          len(runs_full) == 1 and len(runs_split) == 4,
          "all-24 -> " + str(len(runs_full)) + " run, [5,7,9,11] -> "
          + str(len(runs_split)) + " runs")

    covered = sum(hi - lo for lo, hi in runs_full)
    check("merged runs cover the whole intermediate axis exactly once",
          covered == rc.intermediate and runs_full[0] == (0, rc.intermediate),
          "rows=" + str(covered))

    # -- executed-FLOP accounting ------------------------------------------
    per = rc.group_flops(1, 0)
    check("active_flops is the sum of the true per-group costs",
          rc.active_flops(list(range(24))) == 24 * per
          and rc.active_flops([1, 5]) == 2 * per
          and rc.active_flops([]) == 0,
          "one group = " + format(per, ",") + " MACs/token")

    # An uneven partition must not under-report the remainder group.
    uneven = QuantizedMlpRemnant(
        toy_mlp(hidden=64, inter=100), RemnantConfig(0, 7, QuantSpec(4, 128, False),
                                                     torch.float32))
    sizes = [hi - lo for lo, hi in uneven.bounds]
    check("an uneven partition folds the remainder into the last group and "
          "reports its true size", sum(sizes) == 100 and sizes[-1] == 16,
          "sizes=" + str(sizes))
    check("active_flops over an uneven partition equals the whole layer",
          uneven.active_flops(list(range(7))) == 2 * 100 * 64 * 3,
          format(uneven.active_flops(list(range(7))), ","))


# ==========================================================================
# C. Alpha calibration
# ==========================================================================
def verify_alpha() -> None:
    section("C. Output-gain calibration")
    torch.manual_seed(2)
    x = torch.randn(64, 256) * 0.7

    scale_invariant = True
    improves = True
    values = []
    for scale in (0.02, 1.0, 50.0):
        mlp = toy_mlp(scale=scale)
        r = QuantizedMlpRemnant(mlp, RemnantConfig(0, 24, QuantSpec(4, 128, False),
                                                   torch.float32))
        tgt = mlp(x)
        base = r.forward(x)
        a = calibrate_alpha(r, mlp, x)[0].item()
        closed = ((base * tgt).sum() / base.pow(2).sum()).item()
        values.append(a)
        scale_invariant &= abs(a - closed) < 1e-5
        r.alpha.fill_(a)
        improves &= ((r.forward(x) - tgt).norm().item()
                     <= (base - tgt).norm().item() + 1e-9)
    check("alpha matches the closed-form least-squares solution at every "
          "activation scale", scale_invariant,
          "alphas=" + ", ".join(format(v, ".5f") for v in values))
    check("alpha never increases the reconstruction error", improves)

    # Degenerate remnant: F^Q == 0 identically.
    mlp0 = toy_mlp()
    mlp0.up_proj.weight.data.zero_()
    r0 = QuantizedMlpRemnant(mlp0, RemnantConfig(0, 24, QuantSpec(4, 128, False),
                                                 torch.float32))
    a0 = calibrate_alpha(r0, mlp0, x)[0].item()
    check("a remnant that outputs exactly zero yields alpha=1, not NaN",
          a0 == 1.0, "alpha=" + str(a0))

    mlp = toy_mlp()
    r = QuantizedMlpRemnant(mlp, RemnantConfig(0, 24, QuantSpec(4, 128, False),
                                               torch.float32))
    pg = calibrate_alpha(r, mlp, x, per_group=True)
    check("per-group alpha returns one gain per group and all are finite",
          pg.numel() == 24 and bool(torch.isfinite(pg).all()),
          "min=" + format(pg.min().item(), ".4f")
          + " max=" + format(pg.max().item(), ".4f"))

    # -- regressions for the per-group target bug --------------------------
    # A bit-exact remnant is F^Q == F, so EVERY per-group gain must be exactly
    # 1. The earlier implementation fitted each group against the WHOLE MLP
    # output, which makes alpha_g = 1 + <F_g, sum_{h!=g} F_h>/||F_g||^2. That is
    # a different estimator, not a looser one: it returned gains spanning
    # 0.978-1.024 on a lossless remnant and injected 1.3% relative error into a
    # reconstruction that was exact. This check fails loudly under that form.
    mlp_e = toy_mlp(seed=5)
    exact = QuantizedMlpRemnant(mlp_e, RemnantConfig(0, 8, QuantSpec(16, 0, True),
                                                     torch.float64))
    xe = torch.randn(256, mlp_e.gate_proj.in_features, dtype=torch.float64)
    mlp_e = mlp_e.to(torch.float64)
    pg_exact = calibrate_alpha(exact, mlp_e, xe, per_group=True)
    worst = (pg_exact.to(torch.float64) - 1.0).abs().max().item()
    check("per-group alpha is EXACTLY 1 for a bit-exact remnant "
          "(target is F_g, not F)", worst < 1e-9,
          "max |alpha_g - 1| = " + format(worst, ".2e"))

    sc_exact = calibrate_alpha(exact, mlp_e, xe)[0].item()
    check("scalar alpha is exactly 1 for a bit-exact remnant",
          abs(sc_exact - 1.0) < 1e-9, "alpha=" + format(sc_exact, ".12f"))

    # Reference group outputs must themselves partition the donor exactly, or
    # the target above is not what it claims to be.
    from remnant import reference_group_output                      # noqa: PLC0415
    ref_sum = sum(reference_group_output(mlp_e, xe, *exact.bounds[g])
                  for g in range(exact.n_groups))
    gap = (ref_sum - mlp_e(xe)).abs().max().item()
    check("donor group outputs sum to the donor MLP exactly (float64)",
          gap < 1e-12, "max abs = " + format(gap, ".2e"))

    # Calibrating twice must not compound: the fit neutralises the current gain.
    r_idem = QuantizedMlpRemnant(mlp, RemnantConfig(0, 24, QuantSpec(4, 128, False),
                                                    torch.float32))
    first = calibrate_alpha(r_idem, mlp, x)[0].item()
    r_idem.alpha.fill_(first)
    second = calibrate_alpha(r_idem, mlp, x)[0].item()
    check("alpha calibration is idempotent (does not compound with the "
          "gain already installed)", abs(first - second) < 1e-6,
          "first=" + format(first, ".6f") + " second=" + format(second, ".6f"))
    check("calibration restores the alpha buffer it was called with",
          abs(r_idem.alpha[0].item() - first) < 1e-12)

    # Subset gain: fitted against the same rows of the donor, and must not make
    # the partial reconstruction worse.
    sub = [1, 5, 9]
    a_sub = calibrate_alpha(r_idem, mlp, x, subset=sub)[0].item()
    r_idem.alpha.fill_(1.0)
    ref_sub = sum(reference_group_output(mlp, x, *r_idem.bounds[g]) for g in sub)
    out_sub = r_idem.forward_groups(x, sub)
    e_before = ((out_sub - ref_sub).norm() / ref_sub.norm()).item()
    e_after = ((a_sub * out_sub - ref_sub).norm() / ref_sub.norm()).item()
    check("subset alpha does not increase the partial reconstruction error",
          e_after <= e_before + 1e-9,
          "alpha=" + format(a_sub, ".5f") + "  rel err "
          + format(e_before, ".6f") + " -> " + format(e_after, ".6f"))


# ==========================================================================
# D. Confidence features
# ==========================================================================
def verify_confidence() -> None:
    section("D. Confidence features and calibration")
    v = 32768
    torch.manual_seed(3)

    # A uniform distribution: entropy = log|V| exactly, so neg_entropy = 0.
    uniform = torch.zeros(1, v)
    f = features_from_logits(uniform, v)
    check("uniform logits give neg_entropy 0 and max_prob 1/|V|",
          abs(f["neg_entropy"].item()) < 1e-5
          and abs(f["max_prob"].item() - 1.0 / v) < 1e-9,
          "neg_entropy=" + format(f["neg_entropy"].item(), ".2e")
          + " max_prob=" + format(f["max_prob"].item(), ".3e"))

    # A one-hot distribution: entropy 0, so neg_entropy = 1.
    peaked = torch.full((1, v), -60.0)
    peaked[0, 7] = 60.0
    f = features_from_logits(peaked, v)
    check("a near-one-hot distribution gives neg_entropy 1 and max_prob 1",
          abs(f["neg_entropy"].item() - 1.0) < 1e-5
          and abs(f["max_prob"].item() - 1.0) < 1e-6
          and f["top1"].item() == 7)

    # Entropy must be normalised by log|V|, not by log(observed support).
    two_hot = torch.full((1, v), -60.0)
    two_hot[0, 0] = two_hot[0, 1] = 0.0
    f = features_from_logits(two_hot, v)
    expect = 1.0 - math.log(2.0) / math.log(v)
    check("entropy is normalised by log|V| over the FULL vocabulary",
          abs(f["neg_entropy"].item() - expect) < 1e-4,
          "got " + format(f["neg_entropy"].item(), ".5f")
          + " expected " + format(expect, ".5f"))

    # Feature ranges and monotonicity in peakedness.
    logits = torch.randn(2000, 64)
    sharp = features_from_logits(logits * 8.0, 64)
    flat = features_from_logits(logits * 0.05, 64)
    check("all four features rise with peakedness",
          sharp["max_prob"].mean() > flat["max_prob"].mean()
          and sharp["neg_entropy"].mean() > flat["neg_entropy"].mean()
          and sharp["prob_margin"].mean() > flat["prob_margin"].mean()
          and sharp["logit_margin"].mean() > flat["logit_margin"].mean())
    f = features_from_logits(logits, 64)
    check("probability features stay inside [0,1] and margins are non-negative",
          bool((f["max_prob"] >= 0).all() and (f["max_prob"] <= 1).all()
               and (f["neg_entropy"] >= -1e-6).all()
               and (f["neg_entropy"] <= 1 + 1e-6).all()
               and (f["prob_margin"] >= -1e-6).all()
               and (f["logit_margin"] >= -1e-6).all()))
    check("top1 != top2 always", bool((f["top1"] != f["top2"]).all()))

    # Temperature must not change the argmax, only the spread.
    hot = features_from_logits(logits, 64, temperature=0.5)
    cold = features_from_logits(logits, 64, temperature=3.0)
    check("calibration temperature preserves the top-1 token",
          bool((hot["top1"] == f["top1"]).all() and (cold["top1"] == f["top1"]).all()))
    check("higher temperature lowers confidence, lower temperature raises it",
          cold["max_prob"].mean() < f["max_prob"].mean() < hot["max_prob"].mean())
    try:
        features_from_logits(logits, 64, temperature=0.0)
        check("a non-positive temperature is refused", False)
    except ValueError:
        check("a non-positive temperature is refused", True)

    # Temperature fitting recovers a known distortion, on held-out data.
    torch.manual_seed(4)
    true_t = 2.5
    z = torch.randn(8000, 32)
    y = torch.distributions.Categorical(logits=z).sample()
    fitted = fit_temperature(z * true_t, y)
    check("fit_temperature recovers a synthetic miscalibration",
          abs(fitted - true_t) / true_t < 0.15,
          "fitted=" + format(fitted, ".3f") + " true=" + format(true_t, ".2f"))

    # ECE: a perfectly calibrated signal must score ~0.
    torch.manual_seed(5)
    conf = torch.rand(20000)
    correct = (torch.rand(20000) < conf).float()
    cal = expected_calibration_error(conf, correct)
    bad = expected_calibration_error(conf, (torch.rand(20000) < 0.5).float())
    check("ECE is near zero for a calibrated signal and large for a broken one",
          cal["ece"] < 0.02 and bad["ece"] > 0.1,
          "calibrated=" + format(cal["ece"], ".4f")
          + " broken=" + format(bad["ece"], ".4f"))
    check("ECE skips empty bins rather than scoring them as perfect",
          all(b["n"] > 0 for b in cal["bins"]))

    # JS / stability.
    a = torch.randn(100, 50)
    check("JS(p,p) == 0 and stability(p,p) == 1",
          js_divergence(a, a).abs().max().item() < 1e-6
          and abs(stability(a, a).mean().item() - 1.0) < 1e-6)
    check("JS is symmetric and bounded by log 2",
          torch.allclose(js_divergence(a, a * 3), js_divergence(a * 3, a), atol=1e-6)
          and js_divergence(a, a * 20).max().item() <= math.log(2.0) + 1e-6)

    # The lens must start as an exact identity.
    lens = AlignmentLens(64, rank=8)
    h = torch.randn(5, 64)
    check("a freshly initialised low-rank lens is exactly the identity",
          torch.equal(lens(h), h))
    check("a rank-0 lens has no parameters at all",
          not list(AlignmentLens(64, 0).parameters()))

    # The confidence head must borrow, not copy, the LM head.
    norm = nn.LayerNorm(64)
    head_lin = nn.Linear(64, 128, bias=False)
    head = ConfidenceHead(norm, head_lin, AlignmentLens(64, 0), 128)
    check("ConfidenceHead borrows the model's LM head rather than copying it",
          head.lm_head is head_lin and head.final_norm is norm)

    # -- chunking over positions -------------------------------------------
    # features_from_hidden chunks the position axis so the transient does not
    # scale with prompt length. Two separate claims, and they are not the same
    # claim:
    #
    #   1. The *reduction* is per-position, so splitting it is bit-exact. This
    #      is the part chunking is responsible for and it must be exact.
    #   2. The *projection* is a GEMM, and BLAS picks a different kernel for
    #      [133,64]x[64,512] than for [16,64]x[64,512]. So end-to-end agreement
    #      is at GEMM epsilon, not bit-exact -- and this is the same
    #      shape-dependent accumulation-order effect recorded as N8 for cuBLAS,
    #      reproduced here on the CPU path. It is a property of matrix
    #      multiplication on this stack, not of chunking.
    #
    # What must hold is that the gate's *decisions* do not move: the top-1 and
    # top-2 tokens are identical and the scalar features agree to float32 GEMM
    # epsilon.
    from confidence import features_from_hidden                     # noqa: PLC0415
    big = ConfidenceHead(nn.LayerNorm(64), nn.Linear(64, 512, bias=False),
                         AlignmentLens(64, 0), 512)
    torch.manual_seed(11)
    hs = torch.randn(133, 64)

    with torch.no_grad():
        z = big.logits(hs)
    red_whole = features_from_logits(z, 512)
    red_split = {k: torch.cat([features_from_logits(z[i:i + 16], 512)[k]
                               for i in range(0, 133, 16)], dim=0)
                 for k in red_whole}
    check("the confidence reduction is bit-exact when split over positions",
          all(torch.equal(red_whole[k], red_split[k]) for k in red_whole),
          "133 positions in blocks of 16")

    whole = features_from_hidden(big, hs, chunk=10_000)
    split = features_from_hidden(big, hs, chunk=16)
    check("chunking preserves shape and every feature key",
          set(whole) == set(split)
          and all(split[k].shape[0] == 133 for k in split),
          str(sorted(whole)))
    check("chunking does not change which token the lens predicts",
          bool((whole["top1"] == split["top1"]).all()
               and (whole["top2"] == split["top2"]).all()),
          "0 argmax changes over 133 positions")
    worst = max((whole[k] - split[k]).abs().max().item()
                for k in whole if whole[k].dtype.is_floating_point)
    check("chunked features agree with unchunked to float32 GEMM epsilon",
          worst < 1e-5, "max abs diff = " + format(worst, ".2e")
          + " (shape-dependent GEMM, see N8)")

    # A chunk equal to the position count must take the single-shot path and be
    # bit-identical to it: no empty tail, no spurious concatenation.
    odd = features_from_hidden(big, hs, chunk=133)
    check("a chunk equal to the position count is handled without an empty tail",
          all(torch.equal(odd[k], whole[k]) for k in odd))


# ==========================================================================
# E. Significance / utility
# ==========================================================================
def verify_significance() -> None:
    section("E. Significance: sign convention, exposure, ranking controls")
    torch.manual_seed(6)
    v = 64
    teacher = torch.randn(200, v) * 2.0

    check("KL(p||p) == 0",
          kl_divergence(teacher, teacher).abs().max().item() < 1e-6)
    check("KL is non-negative",
          bool((kl_divergence(teacher, torch.randn(200, v)) >= -1e-6).all()))

    # A candidate closer to the teacher must score positive utility.
    deleted = teacher + torch.randn(200, v) * 3.0
    better = teacher + torch.randn(200, v) * 0.3
    worse = teacher + torch.randn(200, v) * 8.0
    u_better = marginal_utility(teacher, deleted, better)
    u_worse = marginal_utility(teacher, deleted, worse)
    check("utility is positive when the candidate helps, negative when it hurts",
          u_better.mean().item() > 0 > u_worse.mean().item(),
          "helpful=" + format(u_better.mean().item(), ".4f")
          + " harmful=" + format(u_worse.mean().item(), ".4f"))
    check("utility is exactly zero when the candidate equals the deleted path",
          marginal_utility(teacher, deleted, deleted).abs().max().item() < 1e-6)

    # Known ordering must be recovered.
    sig = GroupSignificance(4)
    # Values chosen to be exactly representable in binary, so the assertion
    # tests the aggregation rather than float32-to-float64 promotion of 0.4.
    known = torch.stack([
        torch.full((100,), 0.5), torch.full((100,), 0.125),
        torch.full((100,), -0.25), torch.full((100,), 0.75)], dim=1)
    sig.update(known)
    check("ranking recovers a known utility ordering",
          sig.ranking() == [3, 0, 1, 2], str(sig.ranking()))
    check("mean utility is the arithmetic mean, not the sum",
          sig.mean_utility()[0].item() == 0.5,
          "got " + format(sig.mean_utility()[0].item(), ".6f"))
    check("positive rate is tracked separately from mean utility",
          sig.positive_rate().tolist() == [1.0, 1.0, 0.0, 1.0])

    # Exposure normalisation: a rarely-seen but excellent group must win.
    sig2 = GroupSignificance(2)
    sig2.update(torch.tensor([[0.0625, 0.875]]))         # both seen once
    for _ in range(500):
        # Only group 0 is exposed here, 500 further times, at a weak value.
        sig2.update(torch.tensor([[0.0625]]), groups=[0])
    check("a frequently exposed weak group does not out-rank a rare strong one",
          sig2.ranking()[0] == 1 and sig2.exposure.tolist() == [501.0, 1.0],
          "mean_utility=" + str([round(x, 4) for x in sig2.mean_utility().tolist()])
          + " exposure=" + str(sig2.exposure.tolist()))

    sig3 = GroupSignificance(3)
    try:
        sig3.update(torch.zeros(4, 2))
        check("a group-count mismatch is refused", False)
    except ValueError:
        check("a group-count mismatch is refused", True)

    # Ranking controls.
    base = list(range(24))
    sh = shuffled_ranking(base, seed=7)
    check("shuffled ranking is a permutation with the same members",
          sorted(sh) == base and sh != base)
    check("shuffled ranking is deterministic for a fixed seed",
          shuffled_ranking(base, 7) == sh)
    check("reversed ranking is exactly worst-first",
          reversed_ranking(base) == list(reversed(base)))

    mlp = toy_mlp()
    r = QuantizedMlpRemnant(mlp, RemnantConfig(0, 24, QuantSpec(4, 128, False),
                                               torch.float32))
    mr = magnitude_ranking(r)
    check("magnitude ranking is a valid permutation of the groups",
          sorted(mr) == base)

    # Rank correlation sanity, including ties.
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    check("rank correlation is +1 for identical and -1 for reversed orders",
          abs(rank_correlation(a, a)["spearman"] - 1.0) < 1e-9
          and abs(rank_correlation(a, -a)["spearman"] + 1.0) < 1e-9)
    tied = torch.tensor([1.0, 1.0, 2.0, 2.0])
    rc = rank_correlation(tied, torch.tensor([1.0, 1.0, 5.0, 5.0]))
    check("rank correlation handles exact ties without NaN",
          math.isfinite(rc["spearman"]) and math.isfinite(rc["kendall_tau_b"]),
          "spearman=" + format(rc["spearman"], ".4f")
          + " tau_b=" + format(rc["kendall_tau_b"], ".4f"))


# ==========================================================================
# F. Routing
# ==========================================================================
def verify_routing() -> None:
    section("F. Routing: three-level policy, budgets, selection, accounting")
    n = 24

    p = RoutingPolicy(mode="three_level", tau_full=0.3, tau_skip=0.9,
                      gamma=1.0, n_groups=n)
    check("confidence above tau_skip selects nothing",
          p.select(0.95) == [] and p.budget(0.95) == 0)
    check("confidence below tau_full selects everything",
          len(p.select(0.10)) == n and p.budget(0.10) == n)
    check("the interior region selects a strict subset",
          0 < p.budget(0.6) < n, "k(0.6)=" + str(p.budget(0.6)))
    check("rho is 1 at tau_full and 0 at tau_skip",
          abs(p.rho(0.3) - 1.0) < 1e-12 and abs(p.rho(0.9) - 0.0) < 1e-12)

    mono = all(p.budget(c) >= p.budget(c + 0.02)
               for c in [i / 100 for i in range(0, 98)])
    check("budget is monotonically non-increasing in confidence", mono)

    mid = p.rho(0.6)
    check("rho at the midpoint is 0.5 for gamma=1",
          abs(mid - 0.5) < 1e-9, "rho=" + format(mid, ".6f"))
    hi_g = RoutingPolicy(mode="three_level", tau_full=0.3, tau_skip=0.9,
                         gamma=3.0, n_groups=n)
    lo_g = RoutingPolicy(mode="three_level", tau_full=0.3, tau_skip=0.9,
                         gamma=0.4, n_groups=n)
    check("gamma>1 spends less compute and gamma<1 spends more",
          hi_g.rho(0.6) < mid < lo_g.rho(0.6),
          "g3=" + format(hi_g.rho(0.6), ".4f") + " g1=" + format(mid, ".4f")
          + " g0.4=" + format(lo_g.rho(0.6), ".4f"))

    try:
        RoutingPolicy(mode="three_level", tau_full=0.9, tau_skip=0.3)
        check("tau_full >= tau_skip is refused", False)
    except ValueError:
        check("tau_full >= tau_skip is refused", True)
    for bad in ({"mode": "nope"}, {"selector": "nope"}, {"gamma": 0.0},
                {"fraction": 1.5}, {"ranking": [0, 0, 1]}):
        try:
            RoutingPolicy(n_groups=3, **bad)
            check("invalid policy field is refused: " + str(bad), False)
            break
        except ValueError:
            continue
    else:
        check("every invalid policy field is refused", True)

    check("mode=none never selects a group",
          RoutingPolicy(mode="none", n_groups=n).select(0.0) == [])
    check("mode=all always selects every group",
          len(RoutingPolicy(mode="all", n_groups=n).select(1.0)) == n)

    cw = RoutingPolicy(mode="confidence_whole", tau_skip=0.8, n_groups=n)
    check("confidence_whole is all-or-nothing",
          cw.select(0.9) == [] and len(cw.select(0.7)) == n)

    ff = RoutingPolicy(mode="fixed_fraction", fraction=0.25, selector="contiguous",
                       n_groups=n)
    check("fixed_fraction activates ceil(fraction*n) groups",
          len(ff.select(0.0)) == 6, "k=" + str(len(ff.select(0.0))))

    # Selectors.
    rank = list(range(n))[::-1]
    sp = RoutingPolicy(mode="fixed_fraction", fraction=0.25,
                       selector="static_topk", n_groups=n, ranking=rank)
    check("static_topk follows the supplied ranking",
          sp.select(0.0) == rank[:6], str(sp.select(0.0)))

    rp = RoutingPolicy(mode="fixed_fraction", fraction=0.25, selector="random",
                       n_groups=n, seed=11)
    a1, a2 = rp.select(0.0, token_index=0), rp.select(0.0, token_index=0)
    a3 = rp.select(0.0, token_index=1)
    check("random selection is reproducible per token and varies across tokens",
          a1 == a2 and a1 != a3 and len(a1) == 6)

    util = torch.zeros(n)
    util[5] = 10.0
    util[9] = 5.0
    dyn = RoutingPolicy(mode="fixed_fraction", fraction=2 / n,
                        selector="significance", n_groups=n)
    check("dynamic significance picks the highest-utility groups",
          sorted(dyn.select(0.5, utility=util)) == [5, 9],
          str(sorted(dyn.select(0.5, utility=util))))
    check("dynamic significance falls back to the static ranking when no "
          "per-state utility is supplied",
          dyn.select(0.5) == list(range(n))[:2])

    # With a budget of one group, cost has to change *which* group wins:
    # group 5 has twice the utility of group 9, so it wins at equal cost and
    # must lose once it is 100x more expensive.
    one = 1.0 / n
    flat_cost = RoutingPolicy(mode="fixed_fraction", fraction=one,
                              selector="significance", n_groups=n)
    costs = [1.0] * n
    costs[5] = 100.0
    dyn_cost = RoutingPolicy(mode="fixed_fraction", fraction=one,
                             selector="significance", n_groups=n,
                             group_costs=costs)
    picked_flat = flat_cost.select(0.5, utility=util)
    picked_cost = dyn_cost.select(0.5, utility=util)
    check("cost-aware selection prefers the highest utility at equal cost",
          picked_flat == [5], str(picked_flat))
    check("an expensive group is deselected in favour of a cheaper one",
          picked_cost == [9], str(picked_cost))

    # Trace accounting.
    tr = RoutingTrace(n)
    tr.record(0.95, [])
    tr.record(0.10, list(range(n)))
    tr.record(0.60, list(range(6)))
    s = tr.summary()
    expect_active = (0 + n + 6) / (3 * n)
    check("trace active_fraction is groups executed over groups available",
          abs(s["active_fraction"] - expect_active) < 1e-12,
          "got " + format(s["active_fraction"], ".6f")
          + " expected " + format(expect_active, ".6f"))
    check("trace invocation and skip rates are complementary",
          abs(s["invocation_rate"] + s["skip_rate"] - 1.0) < 1e-12
          and abs(s["invocation_rate"] - 2 / 3) < 1e-12)
    check("trace full-invocation rate counts only fully active tokens",
          abs(s["full_invocation_rate"] - 1 / 3) < 1e-12)

    # -- budget rounding ---------------------------------------------------
    # Callers express an exact k as fraction = k/n. That round-trip overshoots in
    # binary floating point for many k (5/24*24 == 5.000000000000001), and a bare
    # ceil would then execute k+1 groups while the caller recorded k. Every k
    # must survive the round-trip.
    exact_k = True
    offenders = []
    for n in (12, 24, 48, 7):
        for k in range(1, n + 1):
            pol = RoutingPolicy(mode="fixed_fraction", fraction=k / n,
                                selector="static_topk", n_groups=n)
            got = pol.budget(0.0)
            if got != k:
                exact_k = False
                offenders.append((n, k, got))
    check("fraction = k/n recovers exactly k groups for every k "
          "(n = 7, 12, 24, 48)", exact_k,
          "OK for all 91 cases" if exact_k else "failures: " + str(offenders[:5]))

    # A genuine fractional budget must still round up.
    frac = RoutingPolicy(mode="fixed_fraction", fraction=0.21,
                         selector="static_topk", n_groups=24)
    check("a genuine fractional budget still rounds up (0.21 x 24 = 5.04 -> 6)",
          frac.budget(0.0) == 6, "k=" + str(frac.budget(0.0)))

    # -- deterministic ties in dynamic significance ------------------------
    # Quantization makes exact ties common. An unstable sort would make the arm
    # irreproducible for exactly the tokens where the choice is arbitrary.
    tie_pol = RoutingPolicy(mode="fixed_fraction", fraction=0.25,
                            selector="significance", n_groups=8)
    flat_u = torch.zeros(8, dtype=torch.float64)
    picks = {tuple(tie_pol.select(0.5, utility=flat_u)) for _ in range(20)}
    check("all-equal utilities resolve deterministically to the lowest group ids",
          picks == {(0, 1)}, "picked " + str(sorted(picks)))

    partial_tie = torch.tensor([0., 9., 0., 9., 0., 0., 0., 0.],
                               dtype=torch.float64)
    check("tied winners are ordered by group id, and losers are excluded",
          tie_pol.select(0.5, utility=partial_tie) == [1, 3],
          str(tie_pol.select(0.5, utility=partial_tie)))

    # Confidence must not perturb the *ranking*: (1-C)^gamma is a per-token
    # scalar. It previously made every value 0.0 at C=1.0, silently turning the
    # significance selector into the contiguous selector.
    u = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
                     dtype=torch.float64)
    same_pick = all(tie_pol.select(c, utility=u) == tie_pol.select(0.0, utility=u)
                    for c in (0.0, 0.25, 0.5, 0.9, 1.0))
    check("dynamic significance selects the same groups at every confidence "
          "(the (1-C)^gamma factor is a scalar and cannot reorder)", same_pick,
          "picked " + str(tie_pol.select(1.0, utility=u)) + " at C=1.0")

    # -- a utility vector of the wrong width is an error, not a silent slice --
    try:
        tie_pol.select(0.5, utility=torch.zeros(5, dtype=torch.float64))
        check("a wrongly-sized per-state utility vector is refused", False)
    except ValueError:
        check("a wrongly-sized per-state utility vector is refused", True)

    # -- policy capability flags -------------------------------------------
    checks_ok = (
        RoutingPolicy(mode="all", n_groups=4).needs_gate_signal is False
        and RoutingPolicy(mode="none", n_groups=4).needs_gate_signal is False
        and RoutingPolicy(mode="fixed_fraction", fraction=0.5,
                          selector="static_topk", n_groups=4).needs_gate_signal is False
        and RoutingPolicy(mode="three_level", n_groups=4).needs_lens is True
        and RoutingPolicy(mode="three_level", n_groups=4).needs_probe is False
        and RoutingPolicy(mode="three_level", feature="probe_need",
                          n_groups=4).needs_probe is True
        and RoutingPolicy(mode="three_level", feature="probe_need",
                          n_groups=4).needs_lens is False
        and RoutingPolicy(mode="fixed_fraction", fraction=0.5,
                          selector="significance",
                          n_groups=4).needs_group_utility is True
        and RoutingPolicy(mode="fixed_fraction", fraction=0.5,
                          selector="static_topk",
                          n_groups=4).needs_group_utility is False)
    check("policy capability flags identify which monitors an arm must pay for",
          checks_ok)

    check("an unknown gate feature is refused rather than KeyError'd later",
          _raises(lambda: RoutingPolicy(mode="three_level", feature="vibes",
                                        n_groups=4), ValueError))

    # -- the vectorized clustered bootstrap must be exact ------------------
    # eval_forced uses cluster_bootstrap_mean, which computes the same
    # distribution in closed form instead of by resampling index tensors. If it
    # were only approximately equal, every interval in the primary endpoint
    # would be subtly wrong, so equality is asserted rather than assumed.
    from stats import (cluster_bootstrap, cluster_bootstrap_mean,   # noqa: PLC0415
                       cluster_bootstrap_mean_pair, cluster_bootstrap_pair)
    gb = torch.Generator().manual_seed(4)
    cl: list[int] = []
    parts = []
    for c in range(40):
        k = int(torch.randint(2, 12, (1,), generator=gb).item())
        cl += [c] * k
        parts.append(torch.randn(k, generator=gb, dtype=torch.float64))
    vals = torch.cat(parts)
    gen = cluster_bootstrap(vals, cl, lambda x: x.mean().item(), n_boot=300)
    fast = cluster_bootstrap_mean(vals, cl, n_boot=300)
    check("vectorized clustered bootstrap matches the general one exactly",
          abs(gen["point"] - fast["point"]) < 1e-12
          and abs(gen["lo"] - fast["lo"]) < 1e-12
          and abs(gen["hi"] - fast["hi"]) < 1e-12,
          "point/lo/hi all identical over " + str(vals.numel())
          + " values in 40 clusters")

    other = vals + torch.randn(vals.numel(), generator=gb, dtype=torch.float64) * 0.1
    gp = cluster_bootstrap_pair(vals, other, cl, n_boot=300)
    fp = cluster_bootstrap_mean_pair(vals, other, cl, n_boot=300)
    check("vectorized paired clustered bootstrap matches the general one",
          abs(gp["point"] - fp["point"]) < 1e-12
          and abs(gp["lo"] - fp["lo"]) < 1e-12,
          "paired difference formed per position before resampling")

    # A clustered interval must be WIDER than one that pretends positions are
    # independent. If it is not, the clustering is not doing anything.
    flat_boot = cluster_bootstrap_mean(vals, list(range(vals.numel())),
                                       n_boot=300)
    w_cl = fast["hi"] - fast["lo"]
    w_flat = flat_boot["hi"] - flat_boot["lo"]
    check("clustering by prompt widens the interval versus treating positions "
          "as independent", w_cl > w_flat,
          "clustered " + format(w_cl, ".5f") + " vs unclustered "
          + format(w_flat, ".5f"))

    # -- serialization round-trip ------------------------------------------
    pol = RoutingPolicy(mode="three_level", feature="probe_need", tau_skip=0.8,
                        tau_full=0.2, gamma=1.5, selector="significance",
                        n_groups=6, group_costs=[1., 2., 1., 1., 3., 1.])
    d = pol.to_dict()
    check("to_dict serializes every field that changes selection, group_costs "
          "included", d["group_costs"] == [1., 2., 1., 1., 3., 1.]
          and d["feature"] == "probe_need" and d["ranking"] == list(range(6)),
          "keys=" + str(sorted(d)))

    # -- the deployment artifact must reload without the dense donor -------
    # This is the claim that decides whether the method is deployable at all:
    # a remnant that needs the full-precision donor to reload has not compressed
    # anything, it has added a second copy. Tested by saving, deleting the donor
    # entirely, and reloading into a module built from a *different* MLP.
    import io                                                       # noqa: PLC0415
    donor = toy_mlp(seed=21)
    spec4 = QuantSpec(4, 128, False)
    src = QuantizedMlpRemnant(donor, RemnantConfig(7, 8, spec4, torch.float32))
    src.alpha.copy_(torch.linspace(0.9, 1.1, 8))
    xr = torch.randn(48, donor.gate_proj.in_features)
    sel = [0, 3, 4, 7]
    want_full = src.forward(xr).clone()
    want_part = src.forward_groups(xr, sel).clone()

    buf = io.BytesIO()
    torch.save(src.state_dict(), buf)
    artifact_bytes = buf.tell()
    buf.seek(0)
    blob = torch.load(buf, map_location="cpu", weights_only=True)

    # Build the destination from an UNRELATED MLP, so any dependence on the
    # original donor's values shows up as a mismatch rather than passing by
    # coincidence.
    other = toy_mlp(seed=99)
    dst = QuantizedMlpRemnant(other, RemnantConfig(7, 8, spec4, torch.float32))
    before = dst.forward(xr).clone()
    missing = dst.load_state_dict(blob, strict=True)
    del donor

    check("the remnant state_dict reloads strictly, with no missing or "
          "unexpected keys", not missing.missing_keys and not missing.unexpected_keys,
          str(len(blob)) + " tensors, " + format(artifact_bytes, ",") + " B")
    check("a reloaded remnant reproduces the full-remnant output bit-exactly",
          torch.equal(dst.forward(xr), want_full))
    check("a reloaded remnant reproduces a PARTIAL group selection bit-exactly",
          torch.equal(dst.forward_groups(xr, sel), want_part),
          "groups " + str(sel))
    check("the reload actually replaced the destination's own weights",
          not torch.equal(before, want_full),
          "unrelated donor gave a different output before loading")
    check("the calibrated gain survives serialization",
          torch.equal(dst.alpha, src.alpha))
    packed_keys = [k for k in blob if k.endswith("qweight")]
    check("every packed weight in the artifact is uint8, and no full-precision "
          "donor copy is present",
          len(packed_keys) == 3
          and all(blob[k].dtype == torch.uint8 for k in packed_keys)
          and not any(blob[k].dtype in (torch.float32, torch.bfloat16)
                      and blob[k].numel() > 8 * 1024 for k in blob),
          "packed=" + str(len(packed_keys)) + "  dtypes="
          + str(sorted({str(v.dtype) for v in blob.values()})))
    # The artifact must be the accounted bytes plus only the container's own
    # overhead -- torch.save writes a zip with a pickle header and one entry per
    # tensor, which is a few KiB for ten tensors and does NOT scale with the
    # weights. What this rules out is an extra tensor payload: if a dense donor
    # or an optimizer state were riding along, the excess would be a multiple of
    # the packed size rather than a fixed container cost.
    packed_total = src.packed_bytes()["total"]
    overhead = artifact_bytes - packed_total
    check("the serialized artifact is the accounted packed bytes plus only "
          "container overhead",
          0 <= overhead < 8192,
          "artifact=" + format(artifact_bytes, ",") + " B  accounted="
          + format(packed_total, ",") + " B  container overhead="
          + format(overhead, ",") + " B over " + str(len(blob)) + " tensors")


# ==========================================================================
# G. Budget accounting
# ==========================================================================
def verify_budget() -> None:
    section("G. Budget accounting against the real checkpoint config")
    cfg = json.loads((REPO / "artifacts" / "rineya-core-en32k-28L"
                      / "config.json").read_text(encoding="utf-8"))
    shape = ModelShape.from_config(cfg, dtype_bits=16)

    weights = REPO / "artifacts" / "rineya-core-en32k-28L" / "model.safetensors"
    if weights.exists():
        on_disk = weights.stat().st_size
        predicted = shape.total_params() * 2
        err = abs(predicted - on_disk) / on_disk
        check("analytic parameter count matches the safetensors file size",
              err < 0.01,
              "predicted=" + format(predicted, ",") + " on_disk="
              + format(on_disk, ",") + " err=" + format(100 * err, ".3f") + "%")

    st = remnant_storage(shape, bits=4, block=128, symmetric=False)
    check("one MLP layer is a small share of the model",
          0.015 < st["share_of_model_params"] < 0.025,
          "share=" + format(100 * st["share_of_model_params"], ".2f") + "%")
    check("int4 compression against bf16 is between 3.5x and 4x",
          3.5 < st["compression_vs_dense"] < 4.0,
          "x" + format(st["compression_vs_dense"], ".3f"))

    # Quantization must not be described as a parameter reduction.
    st16 = remnant_storage(shape, bits=16, block=128, symmetric=True)
    check("quantization changes bytes, not logical parameter count",
          st["logical_parameters"] == st16["logical_parameters"],
          str(st["logical_parameters"]))

    # The closed form for the confidence overhead: |V| / (3 * d_ff).
    from budget import confidence_overhead
    c = confidence_overhead(shape)
    closed = shape.vocab / (3.0 * shape.intermediate)
    check("exact-confidence cost ratio equals |V|/(3*d_ff) in closed form",
          abs(c["confidence_cost_ratio"] - closed) < 1e-9,
          "ratio=" + format(c["confidence_cost_ratio"], ".4f")
          + " = " + str(shape.vocab) + "/(3*" + str(shape.intermediate) + ")")
    check("the exact-confidence monitor costs MORE than the MLP it gates "
          "on this checkpoint", c["confidence_cost_ratio"] > 1.0,
          format(c["confidence_cost_ratio"], ".3f") + "x")

    # Verify the packed-byte model against the real implementation.
    mlp = toy_mlp(hidden=shape.hidden, inter=shape.intermediate)
    r = QuantizedMlpRemnant(mlp, RemnantConfig(0, 24, QuantSpec(4, 128, False),
                                               torch.bfloat16))
    measured = r.packed_bytes()["total"]
    predicted = st["packed_bytes"] + 24 * 4          # + alpha vector
    check("analytic remnant byte model matches the built module",
          abs(measured - predicted) < 8,
          "measured=" + format(measured, ",") + " predicted="
          + format(int(predicted), ","))


# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    torch.use_deterministic_algorithms(False)
    started = time.time()
    print("CSGQR verification suite   (CPU, deterministic)")
    print("torch " + torch.__version__)

    verify_quantization()
    verify_decomposition()
    verify_alpha()
    verify_confidence()
    verify_significance()
    verify_routing()
    verify_budget()

    failed = [r for r in RESULTS if not r["ok"]]
    print("\n" + "=" * 66)
    print(str(len(RESULTS) - len(failed)) + " / " + str(len(RESULTS))
          + " checks passed in " + format(time.time() - started, ".1f") + "s")
    if failed:
        print("\nFAILED:")
        for r in failed:
            print("  - " + r["name"] + ("   " + r["detail"] if r["detail"] else ""))

    if args.json:
        dest = args.json if args.json.is_absolute() else REPO / args.json
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({
            "torch": torch.__version__,
            "total": len(RESULTS),
            "passed": len(RESULTS) - len(failed),
            "failed": len(failed),
            "elapsed_s": round(time.time() - started, 2),
            "checks": RESULTS,
        }, indent=2) + "\n", encoding="utf-8")
        print("wrote " + str(dest.relative_to(REPO)))

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
