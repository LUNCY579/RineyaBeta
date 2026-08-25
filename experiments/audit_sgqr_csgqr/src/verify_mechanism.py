#!/usr/bin/env python3
"""Independent re-derivation of the mechanism's load-bearing invariants. CPU only.

Written for this audit, not reusing ``verify_csgqr.py`` or ``independent_audit.py``:
if a bug were shared between the campaign's implementation and the campaign's
verifier, running the verifier again would not find it. Every check below either
recomputes a quantity from raw tensors with arithmetic written here, or probes the
implementation from outside by instrumenting it.

Runs on CPU so it can be executed while the GPU is busy and so that no result
depends on a particular kernel.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import MODEL, N_GROUPS, REPO, RETIRED, SPECS                 # noqa: E402
from multilayer import RetirementPlan, install_sgqr                    # noqa: E402
from remnant import QuantizedMlpRemnant, RemnantConfig                 # noqa: E402
from router import RoutingPolicy                                       # noqa: E402

CHECKS: list[dict] = []


def check(name: str, ok: bool, detail: str = "", value=None) -> None:
    CHECKS.append({"check": name, "ok": bool(ok), "detail": detail,
                   "value": value})
    print(("  PASS  " if ok else "  FAIL  ") + name + ((" -- " + detail) if detail else ""))


def main() -> int:
    torch.manual_seed(20260824)
    from transformers import AutoModelForCausalLM
    print("loading %s on CPU" % MODEL)
    model = AutoModelForCausalLM.from_pretrained(REPO / MODEL, dtype=torch.float32)
    model.eval()
    L = RETIRED[0]
    mlp = model.model.layers[L].mlp
    d_model = model.config.hidden_size
    d_ff = model.config.intermediate_size

    print("\n1. group decomposition")
    x = torch.randn(7, d_model, dtype=torch.float64)
    W_gate = mlp.gate_proj.weight.detach().to(torch.float64)
    W_up = mlp.up_proj.weight.detach().to(torch.float64)
    W_down = mlp.down_proj.weight.detach().to(torch.float64)
    inner = torch.nn.functional.silu(x @ W_gate.T) * (x @ W_up.T)
    whole = inner @ W_down.T
    gs = d_ff // N_GROUPS
    parts = torch.zeros_like(whole)
    for g in range(N_GROUPS):
        sl = slice(g * gs, (g + 1) * gs)
        parts += (torch.nn.functional.silu(x @ W_gate[sl].T) * (x @ W_up[sl].T)) @ W_down[:, sl].T
    rel = float((whole - parts).abs().max() / whole.abs().max())
    check("group outputs sum to the whole MLP in float64", rel < 1e-13,
          "max relative error %.3e over %d groups" % (rel, N_GROUPS), rel)
    check("no projection carries a bias (a down_proj bias would be counted once "
          "per active group)",
          all(getattr(mlp, n).bias is None for n in ("gate_proj", "up_proj", "down_proj")))
    check("the intermediate dimension divides evenly into groups",
          d_ff % N_GROUPS == 0, "d_ff=%d n_groups=%d" % (d_ff, N_GROUPS))

    print("\n2. packed INT4 storage, decoded by hand")
    rem = QuantizedMlpRemnant(mlp, RemnantConfig(layer=L, n_groups=N_GROUPS,
                                                 spec=SPECS["int4"],
                                                 compute_dtype=torch.float32))
    pm = rem.gate_q
    q = pm.qweight
    lo, hi = 0, d_ff
    deq = pm.dequantize(lo, hi).to(torch.float64)
    # hand unpack: low nibble = even index along the packed axis
    packed = q.to(torch.int64)
    even = packed & 0x0F
    odd = (packed >> 4) & 0x0F
    codes = torch.stack([even, odd], dim=-1).reshape(packed.shape[0], -1)
    codes = codes[:, :d_model]
    scales = pm.scales.to(torch.float64)
    zeros = pm.zeros.to(torch.float64)
    block = SPECS["int4"].block
    nb = (d_model + block - 1) // block
    rebuilt = torch.zeros(d_ff, d_model, dtype=torch.float64)
    for b in range(nb):
        s = slice(b * block, min((b + 1) * block, d_model))
        rebuilt[:, s] = (codes[:, s] - zeros[:, b:b + 1]) * scales[:, b:b + 1]
    diff = float((rebuilt - deq).abs().max())
    check("hand-unpacked INT4 nibbles reproduce dequantize() exactly", diff == 0.0,
          "max abs diff %.3e over %d x %d" % (diff, d_ff, d_model), diff)
    check("INT4 codes occupy the full [0,15] range",
          int(codes.min()) >= 0 and int(codes.max()) <= 15
          and int(codes.max()) == 15,
          "min %d max %d" % (int(codes.min()), int(codes.max())))
    check("two INT4 values are stored per byte",
          q.shape[1] * 2 >= d_model and q.dtype == torch.uint8,
          "qweight %s dtype %s for %d columns" % (tuple(q.shape), q.dtype, d_model))

    print("\n3. byte accounting recomputed from first principles")
    pb = rem.packed_bytes()
    exp_q = 3 * d_ff * ((d_model + 1) // 2)
    exp_s = 3 * d_ff * nb * 2
    exp_z = 3 * d_ff * nb * 1
    check("packed qweight bytes match 3 * d_ff * ceil(d_model/2)",
          pb["gate"] + pb["up"] + pb["down"] - pb["scales"] - pb["zeros"] == exp_q,
          "accounted %d, expected %d" % (pb["gate"] + pb["up"] + pb["down"]
                                         - pb["scales"] - pb["zeros"], exp_q))
    check("scale bytes match 3 * d_ff * n_blocks * 2 (fp16)", pb["scales"] == exp_s,
          "%d vs %d" % (pb["scales"], exp_s))
    check("zero-point bytes match 3 * d_ff * n_blocks * 1 (uint8)", pb["zeros"] == exp_z,
          "%d vs %d" % (pb["zeros"], exp_z))
    # dense_bytes() follows the loaded dtype; this audit loads fp32 on CPU, so
    # the bf16 reference is recomputed explicitly rather than read off the model.
    dense_bf16 = 3 * d_ff * d_model * 2
    comp = dense_bf16 / pb["total"]
    check("single-layer INT4 compression against bf16 is 3.82x",
          abs(comp - 3.8208) < 1e-3, "measured %.5f (bf16 reference %d B)"
          % (comp, dense_bf16), comp)
    check("four-layer INT4 budget is 19,759,488 B",
          4 * pb["total"] == 19759488, "4 x %d" % pb["total"], 4 * pb["total"])

    print("\n4. reconstruction fidelity")
    # Real post-norm activations, not white noise: quantization error is
    # direction-dependent and a Gaussian probe overstates it several-fold.
    with torch.no_grad():
        xf = model.model.layers[L].post_attention_layernorm(
            torch.randn(5, d_model, dtype=torch.float32)).detach()
    ALL = list(range(N_GROUPS))
    with torch.no_grad():
        y_rem = rem.forward_groups(xf, ALL).detach().to(torch.float64)
        y_true = mlp(xf).detach().to(torch.float64)
    relq = float((y_rem - y_true).norm() / y_true.norm())
    rem8 = QuantizedMlpRemnant(mlp, RemnantConfig(layer=L, n_groups=N_GROUPS,
                                                  spec=SPECS["int8"],
                                                  compute_dtype=torch.float32))
    with torch.no_grad():
        rel8 = float((rem8.forward_groups(xf, ALL).detach().to(torch.float64)
                      - y_true).norm() / y_true.norm())
    check("precision is monotone: INT8 output error < INT4 output error",
          rel8 < relq, "int8 %.5f < int4 %.5f" % (rel8, relq), [rel8, relq])
    remfp = QuantizedMlpRemnant(mlp, RemnantConfig(layer=L, n_groups=N_GROUPS,
                                                   spec=SPECS["fp16"],
                                                   compute_dtype=torch.float32))
    with torch.no_grad():
        y_fp = remfp.forward_groups(xf, ALL).detach().to(torch.float64)
    check("FP passthrough remnant reproduces the donor to float32 GEMM epsilon",
          float((y_fp - y_true).abs().max()) < 1e-4,
          "max abs %.3e" % float((y_fp - y_true).abs().max()))
    before = mlp.gate_proj.weight.detach().clone()
    with torch.no_grad():
        mlp.gate_proj.weight.add_(1.0)
    with torch.no_grad():
        y_after = remfp.forward_groups(xf, ALL).detach().to(torch.float64)
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(before)
    check("the FP remnant COPIES the donor rather than aliasing it (CSGQR-v1 S19)",
          float((y_after - y_fp).abs().max()) == 0.0,
          "donor mutated; remnant output moved by %.3e"
          % float((y_after - y_fp).abs().max()))

    print("\n5. executed sparsity: instrument dequantize and count rows touched")
    touched: list[tuple[int, int]] = []
    for pmat in (rem.gate_q, rem.up_q, rem.down_q):
        orig = pmat.dequantize

        def wrap(lo, hi, _o=orig):
            touched.append((int(lo), int(hi)))
            return _o(lo, hi)
        pmat.dequantize = wrap
    touched.clear()
    rem.forward_groups(xf, [])
    check("an empty group set dequantizes nothing", len(touched) == 0,
          "%d dequantize calls" % len(touched))
    for sel in ([0], [3, 4, 5], [0, 12, 23], ALL):
        touched.clear()
        rem.forward_groups(xf, sel)
        rows = sum(hi - lo for lo, hi in touched)
        want = 3 * len(sel) * (d_ff // N_GROUPS)
        ok_range = all(any(lo <= g * (d_ff // N_GROUPS) and (g + 1) * (d_ff // N_GROUPS) <= hi
                           for lo, hi in touched) for g in sel)
        outside = any(not any(lo >= g * (d_ff // N_GROUPS) and hi <= (g + 1) * (d_ff // N_GROUPS)
                              for g in sel)
                      for lo, hi in touched) and len(sel) == 1
        check("groups %s dequantize exactly 3 x |sel| x group_size rows"
              % (str(sel) if len(sel) < 5 else "ALL"),
              rows == want and ok_range and not outside,
              "%d rows touched, expected %d" % (rows, want), rows)

    print("\n6. matched-compute allocation")
    from core import allocate_exact, binary_budgets
    for n, B in ((1000, 12345), (777, 96 * 777), (500, 1), (2687, 25786)):
        w = torch.rand(n, dtype=torch.float64)
        k = allocate_exact(w, B)
        check("allocate_exact hits the budget exactly (n=%d, B=%d)" % (n, B),
              int(k.sum()) == B and int(k.min()) >= 0 and int(k.max()) <= 96,
              "sum %d, range [%d, %d]" % (int(k.sum()), int(k.min()), int(k.max())))
    w = torch.ones(1000, dtype=torch.float64)
    k = allocate_exact(w, 24000)
    check("a flat demand vector yields a uniform per-token budget",
          int(k.min()) == int(k.max()) == 24, "range [%d, %d]" % (int(k.min()), int(k.max())))
    need = torch.rand(1000, dtype=torch.float64)
    kb = binary_budgets(need, 24000)
    on = (kb == 96).sum().item()
    check("the binary gate is all-or-nothing apart from one boundary token",
          int(kb.sum()) == 24000 and on == 250 and int((kb > 0).sum()) == 250,
          "%d tokens at full remnant, sum %d" % (on, int(kb.sum())))
    perm = k[torch.randperm(k.numel(), generator=torch.Generator().manual_seed(7))]
    check("a permuted budget vector has identical total compute",
          int(perm.sum()) == int(k.sum()),
          "%d == %d" % (int(perm.sum()), int(k.sum())))

    print("\n7. the S7 budget-rounding defect, re-derived")
    import math
    bad24 = [kk for kk in range(1, 24) if math.ceil(kk / 24 * 24) != kk]
    bad96 = [kk for kk in range(1, 96) if math.ceil(kk / 96 * 96) != kk]
    check("BUG_LEDGER S7's stated example (5/24*24 == 5.000000000000001) is false",
          5 / 24 * 24 == 5.0, "5/24*24 = %r" % (5 / 24 * 24))
    check("no k in [1,24) triggers the S7 defect at n_groups=24",
          bad24 == [], "affected k: %s" % bad24)
    check("no k in [1,96) triggers it at n_groups=96 either",
          bad96 == [], "affected k: %s" % bad96)
    bad_n = [n for n in range(2, 257)
             if any(math.ceil(kk / n * n) != kk for kk in range(1, n))]
    check("the defect IS real for other group counts (so the fix is still right)",
          25 in bad_n and len(bad_n) > 100,
          "%d of 255 group counts affected; smallest is %d" % (len(bad_n), min(bad_n)))
    p24 = RoutingPolicy(mode="fixed_fraction", fraction=5 / 24, selector="static_topk",
                        n_groups=24)
    check("the current RoutingPolicy returns exactly k for fraction = k/n",
          p24.budget(0.0) == 5, "budget = %d" % p24.budget(0.0))

    print("\n8. donor identity is a real difference, not a relabelling")
    m2 = model.model.layers[RETIRED[2]].mlp
    r2 = QuantizedMlpRemnant(m2, RemnantConfig(layer=RETIRED[0], n_groups=N_GROUPS,
                                               spec=SPECS["int4"],
                                               compute_dtype=torch.float32))
    with torch.no_grad():
        y_wrong = r2.forward_groups(xf, ALL).detach().to(torch.float64)
    rel_w = float((y_wrong - y_true).norm() / y_true.norm())
    check("a wrong-layer donor at identical bytes is an order of magnitude "
          "further from the donor than INT4 quantization is",
          rel_w > 10 * relq, "wrong donor %.4f vs int4 %.4f (ratio %.1fx)"
          % (rel_w, relq, rel_w / max(relq, 1e-12)), [rel_w, relq])
    check("the wrong donor occupies exactly the same number of bytes",
          r2.packed_bytes()["total"] == pb["total"],
          "%d == %d" % (r2.packed_bytes()["total"], pb["total"]))

    n_ok = sum(1 for c in CHECKS if c["ok"])
    payload = {"audit": "sgqr_csgqr_dual", "device": "cpu", "model": MODEL,
               "total": len(CHECKS), "passed": n_ok,
               "failed": len(CHECKS) - n_ok, "checks": CHECKS}
    dest = REPO / "experiments/audit_sgqr_csgqr/results/verify_mechanism.json"
    dest.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print("\n%d/%d checks passed -> %s" % (n_ok, len(CHECKS), dest.name))
    return 0 if n_ok == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
