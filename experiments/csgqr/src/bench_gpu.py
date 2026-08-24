#!/usr/bin/env python3
"""Gate E: measured GPU memory and latency per arm. No inferred numbers.

Rules this script follows, because breaking any of them turns a systems claim
into a fiction:

* Every timed region is bracketed by ``torch.cuda.synchronize()``. CUDA launches
  are asynchronous, so unsynchronized Python wall time measures queueing, not
  execution.
* Kernels are warmed before timing. The first call into a new shape pays cuBLAS
  autotuning and allocator growth, and on this card that first call can be
  several times the steady-state cost.
* Resident remnant bytes are read from ``torch.cuda.memory_allocated`` deltas
  around construction, not from the analytic byte model. The two are compared,
  and a disagreement is reported rather than reconciled.
* Peak *allocated* and peak *reserved* are both reported. Reserved is what the
  card is actually denied to other processes; allocated is what the model asked
  for. A claim about VRAM means reserved.
* ``nvidia-smi`` is sampled alongside as an independent cross-check on the torch
  view.

What this measurement cannot show, stated up front
--------------------------------------------------
One MLP layer is 1.99% of this model. INT4 on it saves 1.47% of model VRAM and
routing caps out at 1.99% of model MACs (``results/budget.json``). Those are
below run-to-run allocator and clock variation on a laptop GPU. So the honest
output of this script is not "CSGQR is faster"; it is a measurement of the
*direction and magnitude* of the per-layer cost, from which the multi-layer case
can be reasoned about. The per-arm remnant-only microbenchmark is included for
exactly that reason: it isolates the remnant from the 98% of the model that every
arm shares, where a real effect is visible.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from envinfo import collect, require_cuda                        # noqa: E402
from quant import QuantSpec                                      # noqa: E402
from remnant import QuantizedMlpRemnant, RemnantConfig           # noqa: E402
from router import RoutingTrace                                  # noqa: E402
from surgery import install_csgqr                                # noqa: E402
from variants import CATALOGUE                                   # noqa: E402

import scripts.train.memory as mem                               # noqa: E402

MODEL = "artifacts/rineya-core-en32k-28L"
MIB = 1 << 20


def smi_used_mib() -> float | None:
    try:
        out = subprocess.run(
            ("nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"),
            capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return float(out.stdout.strip().splitlines()[0])


def sync() -> None:
    torch.cuda.synchronize()


def timed(fn, *, warmup: int, iters: int) -> dict[str, float]:
    """Median/percentile timing of ``fn`` with warmup and explicit sync."""
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    n = len(samples)
    return {
        "mean_s": sum(samples) / n,
        "p50_s": samples[n // 2],
        "p90_s": samples[min(n - 1, int(0.90 * n))],
        "p95_s": samples[min(n - 1, int(0.95 * n))],
        "min_s": samples[0],
        "max_s": samples[-1],
        "stdev_s": statistics.stdev(samples) if n > 1 else 0.0,
        "iters": n,
    }


# --------------------------------------------------------------------------
# Microbenchmark: the remnant alone, isolated from the 98% every arm shares
# --------------------------------------------------------------------------
def bench_remnant_only(donor_mlp, *, tokens: int, n_groups: int,
                       warmup: int, iters: int) -> dict:
    """Cost of the MLP sub-block alone, per arm and per active fraction.

    This is where a routing effect is actually observable: the whole-model
    numbers are dominated by the 27 untouched layers, so a 1.5% change sits
    inside their variance. Reported as a separate table, and never presented as
    an end-to-end speedup.
    """
    device = donor_mlp.gate_proj.weight.device
    x = torch.randn(tokens, donor_mlp.gate_proj.weight.shape[1],
                    dtype=torch.bfloat16, device=device)
    out: dict[str, dict] = {}

    out["dense_nn_linear"] = timed(lambda: donor_mlp(x), warmup=warmup, iters=iters)

    for name, spec in (("fp", QuantSpec(16, 0, True)),
                       ("int8", QuantSpec(8, 128, False)),
                       ("int4", QuantSpec(4, 128, False))):
        r = QuantizedMlpRemnant(donor_mlp, RemnantConfig(0, n_groups, spec,
                                                         torch.bfloat16)).to(device)
        allg = list(range(n_groups))
        out[name + "_all"] = timed(lambda: r.forward_groups(x, allg),
                                   warmup=warmup, iters=iters)
        for frac in (0.125, 0.25, 0.5, 0.75):
            k = max(1, int(round(frac * n_groups)))
            sel = allg[:k]
            out[name + "_rho" + format(frac, ".3f")] = timed(
                lambda s=sel: r.forward_groups(x, s), warmup=warmup, iters=iters)
        del r
        gc.collect()
        torch.cuda.empty_cache()
    return out


def bench_confidence_projection(model, *, tokens: int, warmup: int,
                                iters: int) -> dict:
    """Cost of the exact LM-head confidence projection, measured.

    Compared against the measured MLP cost, this is the empirical version of the
    closed-form ``|V|/(3*d_ff)`` ratio in ``budget.py``. If the measured ratio
    disagrees materially with 3.556, one of the two is wrong and the discrepancy
    is the finding.
    """
    device = next(model.parameters()).device
    h = torch.randn(tokens, model.config.hidden_size,
                    dtype=torch.bfloat16, device=device)
    norm, lm = model.model.norm, model.lm_head

    def project() -> None:
        lm(norm(h))

    def project_and_reduce() -> None:
        z = lm(norm(h)).float()
        lp = torch.log_softmax(z, dim=-1)
        p = lp.exp()
        _ = (-(p * lp).sum(-1), torch.topk(z, 2, dim=-1))

    return {
        "projection_only": timed(project, warmup=warmup, iters=iters),
        "projection_plus_features": timed(project_and_reduce, warmup=warmup,
                                          iters=iters),
        "logits_bytes": tokens * model.config.vocab_size * 2,
    }


def bench_probe_monitors(model, *, tokens: int, n_groups: int,
                         warmup: int, iters: int) -> dict:
    """Measured cost of the probe monitors, against the exact lens.

    This is the comparison the whole deployment argument rests on, and until now
    it existed only as a MAC ratio. The closed form says the exact lens costs
    ``|V|/(3 d_ff) = 3.556x`` the MLP it gates while a linear need probe costs
    ``1/(6 d_ff) = 0.0109%`` and a group router ``|G|/(6 d_ff) = 0.2604%``. A MAC
    ratio is not a latency, though: at these shapes the probe is a
    ``[T, 1024] x [1024, 1]`` GEMM, which is so small it may be dominated by
    launch overhead and therefore *worse* than its MAC share suggests, while the
    lens is a large GEMM running at high efficiency. Measuring both against the
    same MLP on the same card is the only way to know, and a probe that is
    cheap in MACs but not in wall-clock would undercut the claim.

    The probes are constructed here with random weights on purpose: this measures
    the *shape* cost, which is all that latency depends on, and it keeps the
    benchmark runnable without a trained artifact.
    """
    device = next(model.parameters()).device
    hidden = int(model.config.hidden_size)
    h = torch.randn(tokens, hidden, dtype=torch.float32, device=device)

    w_need = torch.randn(1, hidden, dtype=torch.float32, device=device)
    b_need = torch.randn(1, dtype=torch.float32, device=device)
    w_router = torch.randn(n_groups, hidden, dtype=torch.float32, device=device)
    b_router = torch.randn(n_groups, dtype=torch.float32, device=device)

    def need() -> None:
        torch.sigmoid(h @ w_need.t() + b_need)

    def router() -> None:
        _ = h @ w_router.t() + b_router

    def both() -> None:
        torch.sigmoid(h @ w_need.t() + b_need)
        _ = h @ w_router.t() + b_router

    return {
        "need_probe": timed(need, warmup=warmup, iters=iters),
        "group_router": timed(router, warmup=warmup, iters=iters),
        "need_plus_router": timed(both, warmup=warmup, iters=iters),
        "need_bytes_fp32": 4 * (hidden + 1),
        "router_bytes_fp32": 4 * (hidden * n_groups + n_groups),
        "tokens": tokens,
    }


# --------------------------------------------------------------------------
def load_model(device: str):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    sync()
    t0 = time.perf_counter()
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(REPO / MODEL,
                                                 dtype=torch.bfloat16).to(device)
    model.config.use_cache = True
    model = model.eval()
    sync()
    load_s = time.perf_counter() - t0
    return model, load_s, (torch.cuda.memory_allocated() - before) / MIB


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--arms", default="b0-dense,b1-delete,b2-fp-all,"
                                      "b3-int8-all,b4-int4-all,b6-static-25,"
                                      "b7-conf-whole,b10-csgqr")
    ap.add_argument("--n-groups", type=int, default=24)
    ap.add_argument("--prompt-lengths", default="32,128,512,2048")
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--micro-tokens", type=int, default=512)
    ap.add_argument("--cap-mib", type=float, default=7000.0)
    ap.add_argument("--out", default="experiments/csgqr/results/gpu_bench.json")
    args = ap.parse_args()

    require_cuda()
    mem.apply_cap(args.cap_mib)
    device = "cuda:0"
    lengths = [int(x) for x in args.prompt_lengths.split(",")]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in CATALOGUE:
            raise SystemExit("unknown arm " + a)

    idle_smi = smi_used_mib()
    print("idle nvidia-smi used: " + str(idle_smi) + " MiB")
    print("layer " + str(args.layer) + "   arms: " + ", ".join(arms))
    print("prompt lengths " + str(lengths) + "   decode "
          + str(args.decode_tokens) + " tokens\n")

    results: dict[str, dict] = {}

    # -- microbenchmarks, once, on a clean model ---------------------------
    model, load_s, load_mib = load_model(device)
    donor = model.model.layers[args.layer].mlp
    print("microbenchmark: remnant alone at " + str(args.micro_tokens) + " tokens")
    micro = bench_remnant_only(donor, tokens=args.micro_tokens,
                               n_groups=args.n_groups, warmup=args.warmup,
                               iters=args.iters)
    for k, v in micro.items():
        print("  {:<26s} {:>9.4f} ms   p90 {:>9.4f} ms".format(
            k, 1000 * v["p50_s"], 1000 * v["p90_s"]))
    conf_cost = bench_confidence_projection(model, tokens=args.micro_tokens,
                                            warmup=args.warmup, iters=args.iters)
    mlp_ms = 1000 * micro["int4_all"]["p50_s"]
    proj_ms = 1000 * conf_cost["projection_plus_features"]["p50_s"]
    print("  {:<26s} {:>9.4f} ms".format("lm_head projection",
                                         1000 * conf_cost["projection_only"]["p50_s"]))
    print("  {:<26s} {:>9.4f} ms".format("projection + features", proj_ms))
    print("  measured confidence/MLP cost ratio: "
          + format(proj_ms / mlp_ms, ".3f") + "x   (closed form predicts 3.556x "
          "against the dense MLP)")

    probe_cost_measured = bench_probe_monitors(
        model, tokens=args.micro_tokens, n_groups=args.n_groups,
        warmup=args.warmup, iters=args.iters)
    dense_ms = 1000 * micro["dense_nn_linear"]["p50_s"]
    print("\n  monitor cost measured against the dense MLP ("
          + format(dense_ms, ".4f") + " ms at " + str(args.micro_tokens)
          + " tokens):")
    for label, key in (("need probe", "need_probe"),
                       ("group router", "group_router"),
                       ("need + router", "need_plus_router"),
                       ("exact lens + features", None)):
        ms = (proj_ms if key is None
              else 1000 * probe_cost_measured[key]["p50_s"])
        print("    {:<24s} {:>9.4f} ms   {:>9.3f}% of the dense MLP".format(
            label, ms, 100 * ms / dense_ms))
    print()

    # -- resident bytes, measured -----------------------------------------
    resident: dict[str, dict] = {}
    for name, spec in (("fp", QuantSpec(16, 0, True)),
                       ("int8", QuantSpec(8, 128, False)),
                       ("int4", QuantSpec(4, 128, False))):
        gc.collect()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        r = QuantizedMlpRemnant(donor, RemnantConfig(0, args.n_groups, spec,
                                                     torch.bfloat16)).to(device)
        sync()
        measured = (torch.cuda.memory_allocated() - base)
        analytic = r.packed_bytes()["total"]
        resident[name] = {
            "measured_bytes": measured, "analytic_bytes": analytic,
            "measured_mib": measured / MIB, "analytic_mib": analytic / MIB,
            "ratio": measured / max(analytic, 1),
        }
        print("resident " + name + ": measured " + format(measured / MIB, ".3f")
              + " MiB vs analytic " + format(analytic / MIB, ".3f")
              + " MiB   ratio " + format(measured / max(analytic, 1), ".4f"))
        del r
    print()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # -- per-arm end-to-end -----------------------------------------------
    for key in arms:
        arm = CATALOGUE[key]
        model, load_s, load_mib = load_model(device)
        trace = None
        install_report = None
        if not arm.dense:
            n_groups = 1 if arm.mode == "none" else args.n_groups
            trace = RoutingTrace(n_groups)
            donor_layer = None if arm.donor_offset is None else \
                (args.layer + arm.donor_offset) % model.config.num_hidden_layers
            before = torch.cuda.memory_allocated()
            inst = install_csgqr(model, args.layer, spec=arm.spec,
                                 n_groups=n_groups,
                                 policy=arm.policy(n_groups=n_groups),
                                 donor_layer=donor_layer, trace=trace,
                                 keep_donor_resident=False)
            sync()
            install_report = inst.report()
            install_report["memory_delta_mib"] = \
                (torch.cuda.memory_allocated() - before) / MIB

        entry: dict = {
            "arm": key, "label": arm.label,
            "cold_load_s": load_s,
            "model_resident_mib": load_mib,
            "install": install_report,
            "prefill": {}, "decode": {},
        }

        for length in lengths:
            ids = torch.randint(0, model.config.vocab_size, (1, length),
                                device=device)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            def prefill() -> None:
                with torch.inference_mode():
                    model(ids)

            t = timed(prefill, warmup=args.warmup, iters=max(5, args.iters // 2))
            entry["prefill"][str(length)] = {
                **t,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
                "smi_used_mib": smi_used_mib(),
                "tokens_per_s": length / t["p50_s"],
            }

            # Decode: first-token latency and steady-state per-token cost.
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            def generate() -> None:
                with torch.inference_mode():
                    model.generate(ids, max_new_tokens=args.decode_tokens,
                                   do_sample=False,
                                   pad_token_id=model.config.eos_token_id)

            gen = timed(generate, warmup=1, iters=max(3, args.iters // 3))
            with torch.inference_mode():
                sync()
                t0 = time.perf_counter()
                model.generate(ids, max_new_tokens=1, do_sample=False,
                               pad_token_id=model.config.eos_token_id)
                sync()
                first_token_s = time.perf_counter() - t0
            entry["decode"][str(length)] = {
                **gen,
                "first_token_s": first_token_s,
                "decode_tokens": args.decode_tokens,
                "per_token_s": (gen["p50_s"] - first_token_s)
                / max(args.decode_tokens - 1, 1),
                "tokens_per_s": args.decode_tokens / gen["p50_s"],
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
                "smi_used_mib": smi_used_mib(),
            }

        if trace is not None:
            entry["routing"] = trace.summary()

        results[key] = entry
        p = entry["prefill"][str(lengths[0])]
        d = entry["decode"][str(lengths[0])]
        print("{:<16s} load {:>5.2f}s  resident {:>7.1f} MiB"
              "  prefill@{} {:>7.2f} ms  decode {:>6.1f} tok/s"
              "  peak_res {:>7.1f} MiB".format(
                  key, load_s, load_mib, lengths[0], 1000 * p["p50_s"],
                  d["tokens_per_s"], d["peak_reserved_mib"]), flush=True)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    out = {
        "layer": args.layer, "n_groups": args.n_groups,
        "prompt_lengths": lengths, "decode_tokens": args.decode_tokens,
        "warmup": args.warmup, "iters": args.iters,
        "micro_tokens": args.micro_tokens,
        "idle_smi_used_mib": idle_smi,
        "remnant_microbenchmark": micro,
        "confidence_projection": conf_cost,
        "measured_confidence_over_mlp_ratio": proj_ms / mlp_ms,
        "probe_monitors": probe_cost_measured,
        "monitor_share_of_dense_mlp": {
            "need_probe": (1000 * probe_cost_measured["need_probe"]["p50_s"]
                           / dense_ms),
            "group_router": (1000 * probe_cost_measured["group_router"]["p50_s"]
                             / dense_ms),
            "need_plus_router": (
                1000 * probe_cost_measured["need_plus_router"]["p50_s"]
                / dense_ms),
            "exact_lens_plus_features": proj_ms / dense_ms,
        },
        "resident_bytes": resident,
        "arms": results,
        "environment": collect(),
    }
    dest = REPO / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\nwrote " + str(dest.relative_to(REPO)))


if __name__ == "__main__":
    main()
