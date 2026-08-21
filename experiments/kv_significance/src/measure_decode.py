#!/usr/bin/env python3
"""Decode throughput against cache size, across a length ladder.

WHY THIS EXISTS. The only decode measurement in this program is a single 16K row
whose no-eviction baseline took 362 SECONDS to prefill against the eviction arm's
2.4 s. A 151x prefill gap is not a compression effect -- that run was spilling to
host RAM -- so its decode number cannot be trusted either. Above 16K decode was
never measured at all: measure_chunked.py records prefill only.

WHAT THE ARITHMETIC SAYS BEFORE ANY MEASUREMENT. Every decode step reads all
904 MiB of weights regardless of cache size, so shrinking the cache is bounded by
Amdahl's law on memory traffic. With KV at 112 KiB/token and a 1024-token budget:

    n        full KV     ceiling on decode speedup
    4,096      448 MiB       1.33x
   16,384    1,792 MiB       2.65x
   32,768    3,584 MiB       4.42x
   65,536    7,168 MiB       7.94x
  131,072   14,336 MiB      15.00x

So "16x smaller cache" was never going to be 16x faster, and the interesting
question is what fraction of the available ceiling is realised -- and whether that
fraction improves with length, where the KV term finally dominates the weights.

THREE ARMS, because the difference between two of them IS the attribution:

  full            no eviction; the cache holds all n tokens
  evict           eviction on, scoring on -- the mechanism as it actually runs
  evict_noscore   identical cache, but recording OFF during decode

`evict` minus `evict_noscore` is the cost of maintaining the significance matrix
during decode, isolated without a profiler. It should be large: `_score_rows`
runs its own einsum over the whole cache, separate from SDPA's internal QK, and
upcasts keys to fp32 -- so the K half of the cache is read roughly three times per
step rather than once.

TWO CONTAMINATION GUARDS, both learned the hard way:

  thermal    this host's own throughput band is 2.17x between cool and 84 C, which
             is wider than most effects worth measuring. Arms are INTERLEAVED
             round-robin rather than run in blocks, so drift hits every arm
             equally, and GPU temperature is recorded around each timed block.
  memory     if reserved memory approaches the cap the allocator starts fighting,
             and on Windows/WDDM it spills to host RAM instead of failing. Every
             row records reserved bytes and the run flags rows above 90% of cap.

Usage:
    python src/measure_decode.py --lengths 4096 16384 32768
    python src/measure_decode.py --lengths 65536 131072 --no-full   # full won't fit
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/attention_shape/src"))

from scripts.kv import significance as sig  # noqa: E402
from scripts.kv.chunked_prefill import (causal_lower_right, chunked_prefill,  # noqa: E402
                                        stored_length)

MODEL = "artifacts/rineya-core-en32k-28L"
ARMS = ("full", "evict", "evict_noscore")


def gpu_temp_c() -> float | None:
    """Temperature now. Recorded around timed blocks, never during one."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def resident_cache_bytes(cache) -> int:
    """De-duplicated by storage pointer -- a view must not count as a saving."""
    seen: dict[int, int] = {}
    for layer in getattr(cache, "layers", []) or []:
        for t in (getattr(layer, "keys", None), getattr(layer, "values", None)):
            if t is not None:
                st = t.untyped_storage()
                seen[st.data_ptr()] = st.nbytes()
    return sum(seen.values())


@torch.no_grad()
def build_cache(model, ids, arm: str, budget: int, chunk: int, window: int,
                recent: int):
    """Prefill and return (cache, n_prefill). Prefill is always chunked.

    Chunking the `full` arm too is deliberate: it bounds the prefill ACTIVATIONS
    without touching the cache, so the arms differ in cache size and nothing else.
    Prefilling `full` in one shot would OOM on the mask long before the cache did,
    and would make the comparison about prefill rather than about decode.
    """
    n = ids.shape[-1]
    sig.reset_scores()
    policy = sig.Policy(budget=budget, sinks=4, recent=recent) if arm != "full" else None
    cache = sig.SignificanceCache(config=model.config, policy=None)
    sig.set_recording(arm != "full", window=window)

    def compress_fn(c, ci):
        sig.compress(c, policy)

    chunked_prefill(model, ids, cache, chunk=chunk, logits_to_keep=1,
                    compress_fn=(compress_fn if policy is not None else None))
    if policy is not None:
        # Compress once more AFTER the final chunk. chunked_prefill compresses
        # only BETWEEN chunks, so without this the cache enters decode holding
        # B + C tokens rather than B -- and at n == chunk it is never compressed
        # at all. Decoding from a cache at budget is the operating condition;
        # anything else measures a budget nobody deploys.
        sig.compress(cache, policy)
    # The mechanism keeps scoring during decode; the isolation arm does not.
    sig.set_recording(arm == "evict", window=window)
    return cache, n


@torch.no_grad()
def decode_steps(model, cache, n_prefill: int, steps: int, device: str) -> list[float]:
    """Per-step latency in ms. Synchronised per step so no work is hidden."""
    tok = torch.tensor([[1]], device=device)
    dtype = next(model.parameters()).dtype
    lat = []
    for i in range(steps):
        stored = stored_length(cache)
        mask = causal_lower_right(1, stored + 1, device, dtype)
        pos = torch.tensor([[n_prefill + i]], device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(tok, past_key_values=cache, position_ids=pos,
                    attention_mask={"full_attention": mask},
                    use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1e3)
        tok = out.logits[:, -1:].argmax(-1)
    return lat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384, 32768])
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--recent", type=int, default=256)
    ap.add_argument("--steps", type=int, default=48, help="timed decode steps")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3,
                    help="interleaved repetitions; arms rotate within each round")
    ap.add_argument("--no-full", action="store_true",
                    help="skip the uncompressed arm (it does not fit above ~32K)")
    ap.add_argument("--cap-mib", type=float, default=7400.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=EXP / "results" / "measure_decode.json")
    args = ap.parse_args()

    from compat import force_repeat_kv
    from transformers import AutoModelForCausalLM
    force_repeat_kv(verbose=False)
    sig.patch_attention()
    tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    torch.cuda.set_per_process_memory_fraction(min(args.cap_mib, tot) / tot)

    model = AutoModelForCausalLM.from_pretrained(
        REPO / MODEL, dtype=torch.bfloat16).to(args.device).eval()

    arms = [a for a in ARMS if not (args.no_full and a == "full")]
    print(f"decode throughput, budget {args.budget}, chunk {args.chunk}, "
          f"recent {args.recent}")
    print(f"{args.rounds} interleaved rounds x {args.steps} steps "
          f"(+{args.warmup} warmup), cap {args.cap_mib:.0f} MiB of {tot:.0f}\n")
    print(f"{'n':>8} {'arm':<14}{'ms/tok':>9}{'tok/s':>8}{'cache MiB':>11}"
          f"{'resv MiB':>10}{'temp C':>8}")

    # samples[(n, arm)] -> list of per-round median latencies
    samples: dict[tuple[int, str], list[float]] = {}
    meta: dict[tuple[int, str], dict] = {}

    g = torch.Generator(device="cpu").manual_seed(11)
    prompts = {n: torch.randint(0, model.config.vocab_size, (1, n),
                                generator=g).to(args.device) for n in args.lengths}

    for rnd in range(args.rounds):
        for n in args.lengths:
            # ROTATE arm order every round. The first run of this harness kept a
            # fixed order and every arm got monotonically slower across rounds --
            # thermal drift of 20-35%. With a fixed order the last arm is always
            # measured hottest, which is a systematic bias, not noise: it inflated
            # evict_noscore and so UNDERSTATED the scoring cost it exists to
            # isolate. Rotation gives every arm every slot.
            order = arms[rnd % len(arms):] + arms[:rnd % len(arms)]
            for arm in order:
                key = (n, arm)
                if meta.get(key, {}).get("oom"):
                    continue          # do not retry an arm that cannot fit
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                t_before = gpu_temp_c()
                try:
                    cache, npre = build_cache(model, prompts[n], arm, args.budget,
                                              args.chunk, args.window, args.recent)
                    decode_steps(model, cache, npre, args.warmup, args.device)
                    lat = decode_steps(model, cache, npre + args.warmup,
                                       args.steps, args.device)
                except torch.cuda.OutOfMemoryError:
                    meta[key] = {"oom": True}
                    sig.set_recording(False)
                    torch.cuda.empty_cache()
                    if rnd == 0:
                        print(f"{n:>8,} {arm:<14}{'--':>9}{'--':>8}"
                              f"{'--':>11}{'--':>10}{'OOM':>8}")
                    continue
                med = statistics.median(lat)
                samples.setdefault(key, []).append(med)
                meta.setdefault(key, {}).update({
                    "cache_resident_mib": round(resident_cache_bytes(cache) / 2 ** 20, 1),
                    "stored_kv": stored_length(cache),
                    "reserved_mib": round(torch.cuda.max_memory_reserved() / 2 ** 20, 1),
                    "temp_c_before": t_before,
                    "temp_c_after": gpu_temp_c(),
                    "oom": False,
                })
                sig.set_recording(False)
                del cache
                torch.cuda.empty_cache()

    rows = []
    for n in args.lengths:
        for arm in arms:
            key = (n, arm)
            m = meta.get(key, {})
            if m.get("oom") or key not in samples:
                rows.append({"length": n, "arm": arm, "oom": True})
                continue
            s = samples[key]
            # MINIMUM across rounds, not median. Under monotone thermal drift the
            # median is a sample of how hot the card happened to be; the minimum
            # is the least-contended observation of the same work, which is the
            # standard statistic for latency and the only one that is comparable
            # across arms measured at different temperatures.
            best = min(s)
            med = statistics.median(s)
            row = {"length": n, "arm": arm, "oom": False,
                   "ms_per_token": round(best, 3),
                   "ms_per_token_median": round(med, 3),
                   "tok_per_s": round(1000.0 / best, 2),
                   "round_medians_ms": [round(x, 3) for x in s],
                   "round_spread_pct": round(100 * (max(s) - min(s)) / best, 1),
                   "budget": 0 if arm == "full" else args.budget,
                   **{k: v for k, v in m.items() if k != "oom"}}
            row["pressure_pct_of_cap"] = round(
                100 * row["reserved_mib"] / args.cap_mib, 1)
            rows.append(row)
            flag = ""
            if row["pressure_pct_of_cap"] > 90:
                flag = "  <-- MEMORY PRESSURE, treat as contaminated"
            if row["round_spread_pct"] > 25:
                flag += "  <-- UNSTABLE across rounds"
            print(f"{n:>8,} {arm:<14}{row['ms_per_token']:>9.2f}"
                  f"{row['tok_per_s']:>8.1f}{row['cache_resident_mib']:>11.1f}"
                  f"{row['reserved_mib']:>10.0f}"
                  f"{(row['temp_c_after'] if row['temp_c_after'] is not None else -1):>8.0f}"
                  f"{flag}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"model": MODEL, "budget": args.budget, "chunk": args.chunk,
         "recent": args.recent, "window": args.window, "steps": args.steps,
         "warmup": args.warmup, "rounds": args.rounds, "cap_mib": args.cap_mib,
         "device_total_mib": round(tot, 1), "results": rows},
        indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")

    # --- what the numbers mean, computed rather than asserted ----------------
    W = 904.0
    print(f"\n{'n':>8} {'measured':>10} {'ceiling':>9} {'realised':>10}"
          f" {'scoring cost':>14}")
    for n in args.lengths:
        got = {a: next((r for r in rows if r["length"] == n and r["arm"] == a
                        and not r["oom"]), None) for a in ARMS}
        if not got["evict"]:
            continue
        line = f"{n:>8,}"
        if got["full"]:
            speedup = got["full"]["ms_per_token"] / got["evict"]["ms_per_token"]
            kv_full = got["full"]["cache_resident_mib"]
            kv_ev = got["evict"]["cache_resident_mib"]
            ceil = (kv_full + W) / (kv_ev + W)
            line += f"{speedup:>10.2f}x{ceil:>8.2f}x{100*speedup/ceil:>9.0f}%"
        else:
            line += f"{'--':>10} {'--':>8} {'--':>9}"
        if got["evict_noscore"]:
            cost = (got["evict"]["ms_per_token"]
                    / got["evict_noscore"]["ms_per_token"] - 1) * 100
            line += f"{cost:>13.1f}%"
        print(line)
    print("\n'ceiling' is (full KV + weights) / (evicted KV + weights): the most a")
    print("decode speedup can be when every step must read 904 MiB of weights.")
    print("'scoring cost' is how much slower the mechanism runs than the same cache")
    print("with the significance matrix switched off during decode.")


if __name__ == "__main__":
    main()
