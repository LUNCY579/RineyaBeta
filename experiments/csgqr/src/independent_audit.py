#!/usr/bin/env python3
"""Independent re-derivation of the load-bearing CSGQR claims.

``verify_csgqr.py`` shares helper code with the implementation it verifies. That
is normal for a test suite and it is also a blind spot: if ``_unpack`` were wrong,
a round-trip test written with ``_unpack`` would still pass. This script exists to
close that hole for the claims that would be most damaging to get wrong.

Where a claim depends on a project helper, this file **re-implements the check
from raw tensors** -- manual nibble arithmetic instead of ``_unpack``, a
hand-written gated MLP instead of ``forward_groups``, a counter wrapped around
``dequantize`` instead of trusting that slicing happens.

Run:
    python experiments/csgqr/src/independent_audit.py            # CPU only
    python experiments/csgqr/src/independent_audit.py --gpu      # + real model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

RESULTS: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append({"name": name, "ok": bool(ok), "detail": detail})
    print("  [" + ("PASS" if ok else "FAIL") + "] " + name
          + ("   " + detail if detail else ""))
    return bool(ok)


def section(t: str) -> None:
    print("\n" + t)
    print("-" * len(t))


# ==========================================================================
# A. Is INT4 storage really two values per byte?
# ==========================================================================
def audit_packing() -> None:
    section("A. Packed INT4: decoded from raw bytes, not with the project's _unpack")
    from quant import PackedMatrix, QuantSpec

    torch.manual_seed(0)
    rows, cols = 64, 256
    w = torch.randn(rows, cols, dtype=torch.float32)
    spec = QuantSpec(4, 128, False)
    pm = PackedMatrix(w, spec, compute_dtype=torch.float32)

    qw = pm.qweight
    check("qweight dtype is uint8", qw.dtype == torch.uint8, str(qw.dtype))
    check("qweight holds exactly cols/2 bytes per row (two values per byte)",
          qw.shape == (rows, cols // 2),
          str(tuple(qw.shape)) + " for a " + str((rows, cols)) + " matrix")

    # Manually decode the nibbles and reconstruct, using only arithmetic.
    recon = torch.zeros(rows, cols, dtype=torch.float64)
    scales = pm.scales.to(torch.float64)
    zeros = pm.zeros.to(torch.float64)
    block = pm.block
    for r in range(rows):
        for c in range(cols):
            byte = int(qw[r, c // 2].item())
            nib = (byte & 0x0F) if (c % 2 == 0) else ((byte >> 4) & 0x0F)
            b = c // block
            recon[r, c] = (nib - zeros[r, b].item()) * scales[r, b].item()
    project = pm.dequantize().to(torch.float64)
    err = (recon - project).abs().max().item()
    check("hand-decoded nibbles reproduce dequantize() exactly",
          err == 0.0, "max abs diff = " + format(err, ".3e"))

    # Every stored nibble must be inside [0, 15]; anything larger means the
    # "4-bit" claim is false regardless of the container dtype.
    lo_n = (qw & 0x0F).max().item()
    hi_n = ((qw >> 4) & 0x0F).max().item()
    check("every stored code fits in 4 bits", lo_n <= 15 and hi_n <= 15,
          "max low nibble " + str(lo_n) + ", max high nibble " + str(hi_n))

    # And the byte count must equal the claim.
    total = qw.numel() * qw.element_size()
    check("packed bytes equal rows*cols/2 exactly",
          total == rows * cols // 2,
          format(total, ",") + " B for " + format(rows * cols, ",") + " weights")

    # Reconstruction must actually be lossy at 4 bits -- a "quantizer" that is
    # lossless is not quantizing.
    rel = ((project - w.to(torch.float64)).norm()
           / w.to(torch.float64).norm()).item()
    check("4-bit reconstruction is lossy but close (0 < rel err < 0.15)",
          0.0 < rel < 0.15, "rel_fro = " + format(rel, ".5f"))


# ==========================================================================
# B. Is group skipping executed, or masked?
# ==========================================================================
def audit_executed_sparsity() -> None:
    section("B. Executed sparsity: counting rows actually dequantized")
    from quant import PackedMatrix, QuantSpec
    from remnant import QuantizedMlpRemnant, RemnantConfig
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self, h, i):
            super().__init__()
            self.gate_proj = nn.Linear(h, i, bias=False)
            self.up_proj = nn.Linear(h, i, bias=False)
            self.down_proj = nn.Linear(i, h, bias=False)

        def forward(self, x):
            return self.down_proj(
                torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    torch.manual_seed(1)
    hidden, inter, n_groups = 64, 768, 24
    mlp = MLP(hidden, inter).to(torch.float64)
    rem = QuantizedMlpRemnant(
        mlp, RemnantConfig(0, n_groups, QuantSpec(4, 128, False), torch.float64))
    x = torch.randn(16, hidden, dtype=torch.float64)

    # Wrap dequantize to record exactly which row ranges are touched.
    touched: list[tuple[int, int]] = []
    original = PackedMatrix.dequantize

    def spy(self, row_lo=0, row_hi=None):
        hi = self.rows if row_hi is None else row_hi
        touched.append((row_lo, hi))
        return original(self, row_lo, hi)

    PackedMatrix.dequantize = spy
    try:
        for sel in ([], [3], [0, 1, 2, 3], [5, 7, 9, 11], list(range(n_groups))):
            touched.clear()
            rem.forward_groups(x, sel)
            rows_touched = sum(hi - lo for lo, hi in touched)
            expected = 3 * sum(rem.bounds[g][1] - rem.bounds[g][0] for g in sel)
            check("rows dequantized for groups " + str(sel) + " == 3 x selected",
                  rows_touched == expected,
                  "touched " + str(rows_touched) + ", expected " + str(expected)
                  + ", calls " + str(len(touched)))
            # No touched range may lie outside the selected groups' rows.
            allowed = set()
            for g in sel:
                lo, hi = rem.bounds[g]
                allowed.update(range(lo, hi))
            stray = [r for lo, hi in touched for r in range(lo, hi)
                     if r not in allowed]
            check("no row outside the selected groups is dequantized "
                  + str(sel), not stray,
                  "stray rows: " + str(len(stray)))
    finally:
        PackedMatrix.dequantize = original

    # A masked implementation would give the same OUTPUT; only the count above
    # distinguishes them. Confirm the output is right too, computed by hand.
    sel = [5, 7, 9, 11]
    idx = torch.tensor([r for g in sel for r in range(*rem.bounds[g])],
                       dtype=torch.long)
    wg = rem.gate_q.dequantize()[idx]
    wu = rem.up_q.dequantize()[idx]
    wd = rem.down_q.dequantize()[idx]
    manual = (torch.nn.functional.silu(x @ wg.t()) * (x @ wu.t())) @ wd
    got = rem.forward_groups(x, sel)
    check("partial output equals a hand-built slice of the full weights",
          torch.allclose(manual, got, atol=0, rtol=0),
          "max abs diff " + format((manual - got).abs().max().item(), ".3e"))


# ==========================================================================
# C. Group additivity, recomputed without forward_groups
# ==========================================================================
def audit_additivity() -> None:
    section("C. Additivity: F(x) == sum_g F_g(x), computed from raw weights")
    import torch.nn as nn

    torch.manual_seed(2)
    hidden, inter, n_groups = 64, 768, 24
    Wg = torch.randn(inter, hidden, dtype=torch.float64)
    Wu = torch.randn(inter, hidden, dtype=torch.float64)
    Wd = torch.randn(hidden, inter, dtype=torch.float64)
    x = torch.randn(32, hidden, dtype=torch.float64)

    full = (torch.nn.functional.silu(x @ Wg.t()) * (x @ Wu.t())) @ Wd.t()
    size = inter // n_groups
    parts = torch.zeros_like(full)
    for g in range(n_groups):
        lo, hi = g * size, (g + 1) * size
        parts += ((torch.nn.functional.silu(x @ Wg[lo:hi].t())
                   * (x @ Wu[lo:hi].t())) @ Wd[:, lo:hi].t())
    rel = ((parts - full).norm() / full.norm()).item()
    check("additivity holds in float64 to < 1e-14 (raw-tensor derivation)",
          rel < 1e-14, "rel err = " + format(rel, ".3e"))

    # And it must FAIL with a down_proj bias -- which is why the constructor
    # refuses one. If this does not fail, the additivity argument is vacuous.
    bias = torch.randn(hidden, dtype=torch.float64)
    full_b = full + bias
    parts_b = parts + n_groups * bias
    check("a down_proj bias WOULD break additivity (counted once per group)",
          ((parts_b - full_b).norm() / full_b.norm()).item() > 1e-3,
          "the constructor refuses a biased down_proj for this reason")


# ==========================================================================
# D. Is the donor frozen?
# ==========================================================================
def audit_freezing() -> None:
    section("D. Donor freezing: gradients, parameters, optimizer state")
    from quant import QuantSpec
    from remnant import QuantizedMlpRemnant, RemnantConfig
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self, h, i):
            super().__init__()
            self.gate_proj = nn.Linear(h, i, bias=False)
            self.up_proj = nn.Linear(h, i, bias=False)
            self.down_proj = nn.Linear(i, h, bias=False)

    torch.manual_seed(3)
    rem = QuantizedMlpRemnant(
        MLP(64, 768), RemnantConfig(0, 24, QuantSpec(4, 128, False),
                                    torch.float32))

    params = list(rem.parameters())
    check("the remnant exposes ZERO trainable parameters", not params,
          str(len(params)) + " parameters")

    names = [n for n, _ in rem.named_buffers()]
    check("packed storage is registered as buffers", len(names) >= 6,
          str(len(names)) + " buffers: " + ", ".join(names[:4]) + " ...")

    ints = [b.dtype for _, b in rem.named_buffers() if "qweight" in _]
    check("every qweight buffer is an integer dtype (autograd cannot track it)",
          all(d == torch.uint8 for d in ints), str(set(str(d) for d in ints)))

    # An optimizer built the normal way must find nothing.
    try:
        torch.optim.SGD(rem.parameters(), lr=0.1)
        found = True
    except ValueError:
        found = False
    check("torch.optim.SGD(rem.parameters()) raises 'empty parameter list'",
          not found)

    # And the values must not move under a backward pass through the remnant.
    before = rem.gate_q.qweight.clone()
    x = torch.randn(8, 64, requires_grad=True)
    out = rem.forward_groups(x, [0, 1, 2])
    out.sum().backward()
    check("a backward pass through the remnant leaves packed weights unchanged",
          torch.equal(before, rem.gate_q.qweight))
    check("...and gradient still flows to the INPUT (the remnant is not detached)",
          x.grad is not None and torch.isfinite(x.grad).all()
          and x.grad.abs().sum().item() > 0,
          "grad norm " + format(x.grad.norm().item(), ".4f")
          if x.grad is not None else "no grad")


# ==========================================================================
# E. Headline numbers, re-derived from the committed result files
# ==========================================================================
def audit_committed_numbers() -> None:
    section("E. Headline numbers re-derived from committed JSON")
    res = REPO / "experiments" / "csgqr" / "results"

    forced: dict[str, dict] = {}
    for tag in ("A_baselines", "B_matched", "C_gates", "D_tokengates",
                "E_frontier"):
        p = res / ("forced_heldout_" + tag + ".json")
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for k, v in d["arms"].items():
            forced.setdefault(k, v)
    check("all five held-out batches are committed", len(forced) > 30,
          str(len(forced)) + " distinct arms")

    def kl(k):
        return forced[k]["kl"]["point"] if k in forced else None

    kd, k4 = kl("b1-delete"), kl("b4-int4-all")
    rec = (kd - k4) / kd
    check("INT4 recovery recomputed from raw KLs = 0.963",
          abs(rec - 0.9626) < 0.001,
          "(" + format(kd, ".5f") + " - " + format(k4, ".5f") + ") / "
          + format(kd, ".5f") + " = " + format(rec, ".4f"))

    kw = kl("b13-wrong-layer")
    check("wrong-layer donor is worse than deletion, recomputed",
          kw > kd, format(kw, ".5f") + " > " + format(kd, ".5f")
          + "  ratio vs correct donor " + format(kw / k4, ".1f") + "x")

    # Matched-budget ordering, straight from the file.
    order = [(k, kl(k)) for k in ("b6-static-25", "b9-sig-dyn-25",
                                  "b5-random-25", "b6-magnitude-25",
                                  "b6-reversed-25") if kl(k) is not None]
    ok = (order[0][1] < order[2][1] and order[3][1] > order[2][1]
          and order[4][1] > order[2][1])
    check("at rho=0.25: static < random, and magnitude/reversed > random",
          ok, "  ".join(n + " " + format(v, ".5f") for n, v in order))

    # Every arm claiming a matched budget must actually have it.
    exact = [k for k in ("b5-random-25", "b6-static-25", "b6-magnitude-25",
                         "b6-reversed-25", "b9-sig-dyn-25")
             if k in forced]
    acts = {k: (forced[k].get("routing") or {}).get("active_fraction")
            for k in exact}
    check("every rho=0.25 arm has active_fraction EXACTLY 0.25",
          all(a is not None and abs(a - 0.25) < 1e-9 for a in acts.values()),
          str({k: round(v, 6) for k, v in acts.items()}))

    # H2: gate vs its matched random-token partner.
    for gate, rnd in (("b7-conf-whole", "r-tok-79"),
                      ("b12-probe-whole", "r-tok-68")):
        if gate in forced and rnd in forced:
            ag = (forced[gate].get("routing") or {}).get("active_fraction")
            ar = (forced[rnd].get("routing") or {}).get("active_fraction")
            check("H2 pair " + gate + " is activation-matched to " + rnd,
                  abs(ag - ar) < 0.02,
                  format(ag, ".4f") + " vs " + format(ar, ".4f"))
            check("H2 pair " + gate + " beats it with disjoint intervals",
                  forced[gate]["kl"]["hi"] < forced[rnd]["kl"]["lo"],
                  format(kl(gate), ".5f") + " vs " + format(kl(rnd), ".5f"))

    # The additive-proxy overstatement, recomputed.
    sig = json.loads((res / "trace_L20_int4"
                      / "significance_analysis.json").read_text(encoding="utf-8"))
    prb = json.loads((res / "trace_L20_int4"
                      / "probe_analysis.json").read_text(encoding="utf-8"))
    row6 = next(r for r in sig["topk_table"] if r["k"] == 6)
    pr6 = next(r for r in prb["topk_capture"] if r["k"] == 6)
    ratios = {}
    for name, arm, proxy in (("random", "b5-random-25", row6["random_mean"]),
                             ("static", "b6-static-25", row6["significance"]),
                             ("dynamic", "b9-sig-dyn-25", pr6["probe_hidden"])):
        ratios[name] = proxy / (kd - kl(arm))
    check("the additive proxy overstates true utility for every selection",
          all(v > 1.0 for v in ratios.values()),
          "  ".join(k + " " + format(v, ".2f") + "x" for k, v in ratios.items()))
    check("the overstatement grows with how per-token the selection is",
          ratios["dynamic"] > ratios["static"] > ratios["random"])

    o12 = next(r for r in sig["static_vs_oracle"] if r["k"] == 12)
    check("the proxy ORACLE at k=12 exceeds the true utility of all 24 groups",
          o12["oracle"]["point"] > (kd - k4),
          format(o12["oracle"]["point"], ".5f") + " > " + format(kd - k4, ".5f"))

    # GPU
    gp = res / "gpu_bench.json"
    if gp.exists():
        gb = json.loads(gp.read_text(encoding="utf-8"))
        m = gb["remnant_microbenchmark"]
        r_int4 = m["int4_all"]["p50_s"] / m["dense_nn_linear"]["p50_s"]
        r_fp25 = m["fp_rho0.250"]["p50_s"] / m["dense_nn_linear"]["p50_s"]
        check("INT4 remnant is slower than the dense MLP it replaces",
              r_int4 > 1.0, format(r_int4, ".2f") + "x")
        check("FP remnant at rho=0.25 is FASTER than dense "
              "(so hard skipping does work)", r_fp25 < 1.0,
              format(r_fp25, ".2f") + "x")
        share = gb.get("monitor_share_of_dense_mlp") or {}
        if share:
            check("measured probe share is >100x its MAC share of 0.0109%",
                  share["need_probe"] > 100 * 0.000109,
                  "measured " + format(100 * share["need_probe"], ".2f") + "%")


# ==========================================================================
# F. GPU: the real model, and the probe's reported accuracy
# ==========================================================================
def audit_gpu() -> None:
    section("F. Real checkpoint: bit-exact reconstruction and probe AUROC")
    from envinfo import require_cuda
    require_cuda()
    from quant import QuantSpec
    from remnant import QuantizedMlpRemnant, RemnantConfig
    from transformers import AutoModelForCausalLM

    dev = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(
        REPO / "artifacts/rineya-core-en32k-28L",
        dtype=torch.bfloat16).to(dev).eval()
    mlp = model.model.layers[20].mlp

    # bits=16 must reconstruct the donor BIT-exactly, or every "same weights"
    # comparison in the campaign is invalid.
    fp = QuantizedMlpRemnant(
        mlp, RemnantConfig(20, 24, QuantSpec(16, 0, True), torch.bfloat16)).to(dev)
    ok = (torch.equal(fp.gate_q.dequantize(), mlp.gate_proj.weight)
          and torch.equal(fp.up_q.dequantize(), mlp.up_proj.weight)
          and torch.equal(fp.down_q.dequantize(),
                          mlp.down_proj.weight.t().contiguous()))
    check("bits=16 reconstructs all three donor matrices BIT-exactly", ok)
    check("...and it is a copy, not an alias",
          fp.gate_q.dense.data_ptr() != mlp.gate_proj.weight.data_ptr())

    # Recompute the need probe's held-out AUROC from the saved artifact, using
    # an AUROC written here rather than the project's.
    from probe_runtime import load_need_probe
    from measure_layers import DATASETS, load_prompts
    from trace_utility import capture_residual
    from transformers import AutoTokenizer

    art = REPO / "artifacts/csgqr_probes/need_L20_int4_g24.pt"
    if not art.exists():
        check("need probe artifact exists", False, str(art))
        return
    predict, meta = load_need_probe(art, dev)

    tok = AutoTokenizer.from_pretrained(REPO / "artifacts/rineya-core-en32k-28L")
    model.config.use_cache = False
    prompts = load_prompts(REPO / DATASETS[meta["dataset"]], tok,
                           meta["train_prompts"] + meta["val_prompts"]
                           + meta["test_prompts"], 256)
    rows = [json.loads(l) for l in
            (REPO / meta["trace"] / "traces.jsonl").read_text(
                encoding="utf-8").splitlines() if l.strip()]

    qs, ys, pidx = [], [], []
    for pi, p in enumerate(prompts):
        with torch.no_grad():
            h = capture_residual(model, meta["layer"], p["ids"].to(dev),
                                 p["boundary"])
            qs.append(predict(h).float().cpu())
        pidx.extend([pi] * h.shape[0])
    q = torch.cat(qs).to(torch.float64)
    util = torch.tensor([r["utility"] for r in rows], dtype=torch.float64)
    if q.numel() != util.numel():
        check("probe positions line up with the trace", False,
              str(q.numel()) + " vs " + str(util.numel()))
        return
    y = (util > meta["delta"]).double()
    cut = meta["train_prompts"] + meta["val_prompts"]
    te = torch.tensor(pidx) >= cut

    # AUROC by the Mann-Whitney identity, written here, not imported.
    def auroc_local(score, label):
        s, l = score[te], label[te]
        pos, neg = s[l > 0.5], s[l <= 0.5]
        if not pos.numel() or not neg.numel():
            return float("nan")
        wins = (pos.reshape(-1, 1) > neg.reshape(1, -1)).double().sum()
        ties = (pos.reshape(-1, 1) == neg.reshape(1, -1)).double().sum()
        return ((wins + 0.5 * ties) / (pos.numel() * neg.numel())).item()

    got = auroc_local(q, y)
    claimed = meta["test"]["auroc"]
    check("need-probe held-out AUROC reproduces the reported figure",
          abs(got - claimed) < 0.01,
          "recomputed " + format(got, ".4f") + " vs reported "
          + format(claimed, ".4f") + " on " + str(int(te.sum())) + " positions")
    check("...and it is far above the raw lens's 0.55-0.60", got > 0.85,
          format(got, ".4f"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    print("=" * 70)
    print("Independent audit -- claims re-derived without the project's helpers")
    print("=" * 70)
    audit_packing()
    audit_executed_sparsity()
    audit_additivity()
    audit_freezing()
    audit_committed_numbers()
    if args.gpu:
        audit_gpu()

    failed = [r for r in RESULTS if not r["ok"]]
    print("\n" + "=" * 70)
    print(str(len(RESULTS) - len(failed)) + " / " + str(len(RESULTS))
          + " independent checks passed")
    if failed:
        print("\nFAILED:")
        for r in failed:
            print("  - " + r["name"] + "   " + r["detail"])
    if args.json:
        dest = REPO / args.json
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(
            {"total": len(RESULTS), "failed": len(failed), "checks": RESULTS},
            indent=2) + "\n", encoding="utf-8")
        print("\nwrote " + str(dest.relative_to(REPO)))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
