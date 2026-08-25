#!/usr/bin/env python3
"""Systems audit: latency, throughput, VRAM and the contention that invalidates them.

Three changes from the historical benchmark, each because of a specific weakness
in it:

1. **Arms are interleaved, not run back to back.** CSGQR-v1 ran each arm's twelve
   iterations consecutively on a contended card and then compared medians. A
   contention burst lands entirely inside one arm under that schedule and shows
   up as an algorithmic difference. Here one round measures every arm once, and
   many rounds are run, so a burst is spread across arms instead of attributed to
   one.

2. **Contention is measured, not asserted.** A reference workload is re-timed at
   the start of every round. The spread of that reference across rounds is the
   noise floor of the whole benchmark, and any arm-to-arm difference smaller than
   it is reported as unresolvable rather than as a result.

3. **SGQR and CSGQR run through the SAME module.** In the historical code the
   gated arm carried a per-token Python bucketing loop that the fixed arm did not,
   worth 16x wall-clock, so "CSGQR is slower" measured the loop and not the gate.
   Here both are :class:`core.LevelledSGQRMlp`; the fixed arm simply has one
   distinct level. The gate's own cost is then measured separately as a monitor
   cost, which is what it actually is.

Nothing here is offered as an absolute performance claim. The card is shared with
a desktop compositor and whatever else the machine is doing, and the reference
spread below says how much that matters.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import (N_GROUPS, REPO, RETIRED, SPECS, TOTAL_GROUPS,        # noqa: E402
                  alloc_from_order, env_manifest, global_order, install,
                  load_model, make_head)
from multilayer import RetirementPlan, install_sgqr                    # noqa: E402

import scripts.train.memory as mem                                     # noqa: E402
from envinfo import require_cuda                                       # noqa: E402

RES = REPO / "experiments/audit_sgqr_csgqr/results"


def sync() -> None:
    torch.cuda.synchronize()


def timed(fn, *, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    sync()
    t = time.perf_counter()
    fn()
    sync()
    return time.perf_counter() - t


def smi_used() -> float | None:
    import subprocess
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return float(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def stats(xs: list[float]) -> dict:
    xs = sorted(xs)
    n = len(xs)
    med = st.median(xs)
    return {"n": n, "median_ms": 1000 * med, "mean_ms": 1000 * st.mean(xs),
            "min_ms": 1000 * xs[0], "max_ms": 1000 * xs[-1],
            "p25_ms": 1000 * xs[max(0, int(0.25 * n) - 1)],
            "p75_ms": 1000 * xs[min(n - 1, int(0.75 * n))],
            "iqr_over_median": ((xs[min(n - 1, int(0.75 * n))]
                                 - xs[max(0, int(0.25 * n) - 1)]) / med) if med else None,
            "max_over_min": (xs[-1] / xs[0]) if xs[0] else None,
            "stdev_ms": 1000 * (st.stdev(xs) if n > 1 else 0.0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--prefill", default="32,128,512")
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--cap-mib", type=float, default=6000.0)
    ap.add_argument("--significance",
                    default="experiments/audit_sgqr_csgqr/results/significance_96.json")
    args = ap.parse_args()

    require_cuda()
    mem.apply_cap(args.cap_mib)
    device = "cuda:0"
    torch.manual_seed(20260824)

    sig = json.loads((REPO / args.significance).read_text(encoding="utf-8"))
    order = global_order({int(k): v for k, v in sig["scores"].items()})

    out: dict = {"audit": "sgqr_csgqr_dual", "rounds": args.rounds,
                 "micro_tokens": args.tokens, "batch": args.batch,
                 "retired_layers": list(RETIRED), "n_groups_per_layer": N_GROUPS,
                 "smi_used_mib_at_start": smi_used(),
                 "environment": env_manifest()}

    # =================================================================
    # A. Sub-block microbenchmark, interleaved
    # =================================================================
    model, tok = load_model(device)
    inst = install(model, spec=SPECS["int4"])
    inst8 = None
    d_model = model.config.hidden_size
    x = torch.randn(args.batch, args.tokens, d_model, dtype=torch.bfloat16,
                    device=device)

    layer = RETIRED[0]
    dense_mlp = inst.pristine[layer]
    rem = inst.modules[layer].remnant

    def make_call(groups):
        def f():
            rem.forward_groups(x.reshape(-1, d_model), groups)
        return f

    ALL = list(range(N_GROUPS))
    calls = {"dense_nn_linear": (lambda: dense_mlp(x))}
    for rho, k in ((0.125, 3), (0.25, 6), (0.50, 12), (0.75, 18), (1.0, 24)):
        calls["int4_rho%.3f" % rho] = make_call(ALL[:k])

    # the dequantization half on its own, so the penalty is attributed
    def dequant_only():
        lo, hi = rem.bounds[0][0], rem.bounds[N_GROUPS - 1][1]
        rem.gate_q.dequantize(lo, hi)
        rem.up_q.dequantize(lo, hi)
        rem.down_q.dequantize(lo, hi)
    calls["dequantize_all_groups_only"] = dequant_only

    head = make_head(model)
    hs = torch.randn(args.tokens, d_model, dtype=torch.bfloat16, device=device)
    probe_w = torch.randn(d_model, 1, dtype=torch.float32, device=device)

    def lens_monitor():
        from core import lens_features
        lens_features(head, hs)

    def probe_monitor():
        (hs.float() @ probe_w)
    calls["monitor_need_probe"] = probe_monitor
    calls["monitor_exact_lens_plus_features"] = lens_monitor

    samples = {k: [] for k in calls}
    ref = []
    print("A. sub-block microbenchmark, %d interleaved rounds, %d tokens"
          % (args.rounds, args.tokens))
    for r in range(args.rounds):
        ref.append(timed(calls["dense_nn_linear"], warmup=1))
        for k, fn in calls.items():
            samples[k].append(timed(fn, warmup=1))
    micro = {k: stats(v) for k, v in samples.items()}
    ref_s = stats(ref)
    base = micro["dense_nn_linear"]["median_ms"]
    for k in micro:
        micro[k]["ratio_vs_dense_mlp"] = micro[k]["median_ms"] / base
    out["reference_workload_spread"] = ref_s
    out["remnant_microbenchmark"] = micro
    out["noise_floor_note"] = (
        "the reference dense MLP was re-timed at the start of every round; its "
        "max/min across rounds is the resolution limit of every ratio below")
    print("   reference spread across rounds: median %.4f ms, max/min %.2fx, "
          "IQR/median %.3f" % (ref_s["median_ms"], ref_s["max_over_min"],
                               ref_s["iqr_over_median"]))
    for k in sorted(micro, key=lambda z: micro[z]["median_ms"]):
        print("   %-32s %8.4f ms  %6.3fx  (min %.4f max %.4f)"
              % (k, micro[k]["median_ms"], micro[k]["ratio_vs_dense_mlp"],
                 micro[k]["min_ms"], micro[k]["max_ms"]))

    # =================================================================
    # B. Gate cost isolated: same engine, one level vs many
    # =================================================================
    print("\nB. engine cost of a per-token level vector (SGQR vs CSGQR, same module)")
    mod = inst.modules[layer]
    n_flat = args.batch * args.tokens
    engine = {}
    for nlev in (1, 2, 4, 8, 16, 24):
        ks = [max(1, round(N_GROUPS * i / max(1, nlev - 1))) if nlev > 1 else 12
              for i in range(nlev)]
        ks = [min(N_GROUPS, max(1, k)) for k in ks]
        mod.set_groups_by_level([ALL[:k] for k in ks])
        lv = torch.arange(n_flat) % nlev
        mod.set_levels(lv)
        xs = x.reshape(-1, d_model)
        vals = [timed(lambda: mod(xs), warmup=1) for _ in range(args.rounds)]
        mean_groups = sum(ks[i % nlev] for i in range(n_flat)) / n_flat
        engine["levels_%d" % nlev] = {**stats(vals),
                                      "mean_groups_per_token": mean_groups,
                                      "distinct_levels": nlev}
        print("   %2d distinct levels  mean k=%5.2f  %8.4f ms"
              % (nlev, mean_groups, engine["levels_%d" % nlev]["median_ms"]))
    mod.set_levels(None)
    mod.set_groups_by_level([ALL])
    out["engine_cost_by_level_count"] = engine

    # =================================================================
    # C. End to end: prefill, decode, peak VRAM, donors FREED
    # =================================================================
    del model, inst, x, hs
    gc.collect()
    torch.cuda.empty_cache()

    lengths = [int(v) for v in args.prefill.split(",")]
    ARMS = [("dense", None, None),
            ("delete4", "int4", 0),
            ("sgqr-0.25", "int4", 24),
            ("sgqr-0.50", "int4", 48),
            ("remnant-int4-full", "int4", 96),
            ("remnant-int8-full", "int8", 96)]
    e2e = {}
    print("\nC. end to end (donors freed; peak reserved is this process only)")
    for name, bits, k in ARMS:
        model, tok = load_model(device)
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        if bits is not None:
            plan = RetirementPlan(layers=RETIRED, spec=SPECS[bits],
                                  n_groups=N_GROUPS)
            alloc = alloc_from_order(order, k)
            inner = install_sgqr(model, plan, allocation=alloc,
                                 keep_donor_resident=False)
        after = torch.cuda.memory_allocated()
        resident = after - before
        rec = {"arm": name, "bits": bits, "active_groups": k,
               "model_resident_mib": after / 2 ** 20,
               "install_delta_mib": resident / 2 ** 20,
               "prefill": {}, "decode": {}}
        for T in lengths:
            ids = torch.randint(0, model.config.vocab_size, (args.batch, T),
                                device=device)
            model.config.use_cache = False
            torch.cuda.reset_peak_memory_stats()
            vals = [timed(lambda: model(ids), warmup=1) for _ in range(max(6, args.rounds // 3))]
            rec["prefill"][str(T)] = {**stats(vals),
                                      "tokens_per_s": args.batch * T / st.median(vals),
                                      "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2 ** 20,
                                      "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20}
            del ids
        model.config.use_cache = True
        ids = torch.randint(0, model.config.vocab_size, (args.batch, 32), device=device)
        torch.cuda.reset_peak_memory_stats()

        def gen():
            with torch.no_grad():
                model.generate(ids, max_new_tokens=args.decode_tokens,
                               do_sample=False, use_cache=True,
                               pad_token_id=tok.pad_token_id or 0)
        vals = [timed(gen, warmup=1) for _ in range(max(4, args.rounds // 6))]
        rec["decode"][str(args.decode_tokens)] = {
            **stats(vals),
            "tokens_per_s": args.decode_tokens / st.median(vals),
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2 ** 20,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20}
        rec["smi_used_mib"] = smi_used()
        e2e[name] = rec
        print("   %-20s prefill@32 %8.3f ms   decode %6.2f tok/s   peak_res %7.1f MiB"
              % (name, rec["prefill"][str(lengths[0])]["median_ms"],
                 rec["decode"][str(args.decode_tokens)]["tokens_per_s"],
                 rec["decode"][str(args.decode_tokens)]["peak_reserved_mib"]))
        del model, ids
        gc.collect()
        torch.cuda.empty_cache()
    out["end_to_end"] = e2e

    # =================================================================
    # D. Storage accounting
    # =================================================================
    model, tok = load_model(device)
    store = {}
    for bits in ("fp16", "int8", "int4"):
        gc.collect()
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated()
        plan = RetirementPlan(layers=RETIRED, spec=SPECS[bits], n_groups=N_GROUPS)
        inner = install_sgqr(model, plan, allocation=None, keep_donor_resident=True)
        after = torch.cuda.memory_allocated()
        analytic = sum(m.remnant.packed_bytes()["total"] for m in inner.modules.values())
        dense_b = sum(m.remnant.dense_bytes() for m in inner.modules.values())
        store[bits] = {"measured_bytes": after - before, "analytic_bytes": analytic,
                       "dense_bytes": dense_b,
                       "measured_mib": (after - before) / 2 ** 20,
                       "analytic_mib": analytic / 2 ** 20,
                       "ratio_measured_over_analytic": (after - before) / analytic,
                       "accounted_compression": dense_b / analytic,
                       "measured_compression": dense_b / (after - before)}
        inner.remove()
        del inner
    out["resident_bytes"] = store
    print("\nD. storage (four retired layers)")
    for b, v in store.items():
        print("   %-6s measured %8.3f MiB  analytic %8.3f MiB  ratio %.4f  "
              "compression accounted %.2fx measured %.2fx"
              % (b, v["measured_mib"], v["analytic_mib"],
                 v["ratio_measured_over_analytic"], v["accounted_compression"],
                 v["measured_compression"]))

    out["smi_used_mib_at_end"] = smi_used()
    contended = (ref_s["max_over_min"] or 1.0) > 1.35
    out["contention_verdict"] = {
        "reference_max_over_min": ref_s["max_over_min"],
        "reference_iqr_over_median": ref_s["iqr_over_median"],
        "absolute_latency_usable": not contended,
        "label": ("SHARED_GPU_TIMING -- ABSOLUTE LATENCY NOT USABLE"
                  if contended else "reference spread below 1.35x; ratios usable"),
    }
    (RES / "systems_audit.json").write_text(json.dumps(out, indent=1) + "\n",
                                            encoding="utf-8")
    with (RES / "timing_samples.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["block", "key", "sample_index", "seconds"])
        for k, v in samples.items():
            for i, s in enumerate(v):
                w.writerow(["microbenchmark", k, i, s])
        for i, s in enumerate(ref):
            w.writerow(["reference", "dense_nn_linear", i, s])
    print("\n%s" % out["contention_verdict"]["label"])
    print("wrote systems_audit.json and timing_samples.csv")


if __name__ == "__main__":
    main()
