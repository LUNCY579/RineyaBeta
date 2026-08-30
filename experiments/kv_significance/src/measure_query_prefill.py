#!/usr/bin/env python3
"""O12, final item -- what the QUERY-AWARE PREFILL PASS actually costs. CPU only.

THE CLAIM UNDER TEST is a docstring in `scripts/kv/chunked_prefill.py`:

    "Cost is one forward over `len(query)` tokens per chunk -- at 16 question
     tokens against a 2048-token cache that is negligible beside the chunk
     itself."

That was asserted, never measured. It contains an arithmetic assumption -- that a
forward's cost is proportional to its token count -- which is FALSE at small token
counts, because a short forward is bound by streaming the weights, not by the
arithmetic on the tokens. If the assumption fails, 16/4096 = 0.4% is the wrong
prediction and the true tax could be an order of magnitude larger.

WHY THIS RUNS ON CPU AND TOUCHES NO GPU. The question is a RATIO of two costs
inside one process. A contended GPU measures contention; this machine's GPU is
held by another workload and its thermal band is 2.17x. CPU float32 is quiet,
reproducible, and -- for the structural question, "is the tax proportional to Q or
is there a floor?" -- gives the answer that transfers. The MAGNITUDE is
device-specific and is reported as such, never as a universal constant.

TWO PARTS.

  A. COMPONENT LADDER. Time a forward of T tokens against a fixed cache, for T
     across a ladder, and fit t(T) = a + b*T. `a` is the fixed floor -- the cost
     of touching every weight once. If a >> b*16 the docstring's premise is
     wrong. Also times `_restore`, the cache copy `score_with_query` performs on
     EVERY layer at EVERY boundary, whose cost depends on cache length and not
     on Q at all -- a term the docstring does not mention.

  B. END-TO-END, SELF-INSTRUMENTED. Rather than subtract two arms and hope they
     are matched, the `aware` run reports its own breakdown: chunk forwards,
     query-scoring passes, compressions. The tax is then a within-run ratio with
     no matching assumption. `blind` and `plain` give the two comparisons that
     do need arms, and those ARE matched: blind and aware hold caches of
     identical shape at every boundary (same budget, same keep count), so their
     chunk forwards are comparable.

Arms are interleaved round-robin and the statistic is the MINIMUM across rounds,
the same protocol as measure_decode and measure_select: the minimum is the
least-contended observation of the same work.

    python experiments/kv_significance/src/measure_query_prefill.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Set BEFORE torch is imported. On this Windows build torch still reports
# cuda.is_available() True regardless, so the real guarantee is the assertion at
# the end of main(): every tensor here is CPU, so a CUDA context is never
# created, and torch.cuda.is_initialized() proves it.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.kv import significance as sig  # noqa: E402
from scripts.kv.chunked_prefill import (_restore, _snapshot,  # noqa: E402
                                        causal_lower_right, chunked_prefill,
                                        score_with_query, stored_length)

MODEL = "artifacts/rineya-core-en32k-28L"
DEV = torch.device("cpu")


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:  # noqa: BLE001
        return float("nan")


# --------------------------------------------------------------------------- #
# A. component ladder
# --------------------------------------------------------------------------- #

def prefill_to(model, n: int, dtype):
    """A real cache of length n, built with recording OFF."""
    from transformers.cache_utils import DynamicCache
    c = DynamicCache()
    ids = torch.randint(0, 30000, (1, n), device=DEV)
    mask = causal_lower_right(n, n, DEV, dtype)
    pos = torch.arange(n, device=DEV).unsqueeze(0)
    model(ids, past_key_values=c, position_ids=pos,
          attention_mask={"full_attention": mask}, use_cache=True, logits_to_keep=1)
    return c


def fit_line(xs, ys):
    """Least squares y = a + b x. Returns (a, b, r2)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, (1 - ss_res / ss_tot if ss_tot else 1.0)


def part_a(model, dtype, ladder, cache_len, rounds):
    print("\n  A. forward cost vs token count, cache length "
          f"{cache_len:,} ({dtype}, cpu)")
    print(f"    {'tokens T':>10}{'ms (min)':>12}{'ms/token':>12}")
    cache = prefill_to(model, cache_len, dtype)
    snap = _snapshot(cache)
    samples = {t: [] for t in ladder}
    for _ in range(rounds):
        for T in ladder:                                   # interleaved
            ids = torch.randint(0, 30000, (1, T), device=DEV)
            mask = causal_lower_right(T, cache_len + T, DEV, dtype)
            pos = torch.arange(cache_len, cache_len + T, device=DEV).unsqueeze(0)
            t0 = time.perf_counter()
            model(ids, past_key_values=cache, position_ids=pos,
                  attention_mask={"full_attention": mask},
                  use_cache=True, logits_to_keep=1)
            samples[T].append((time.perf_counter() - t0) * 1000.0)
            _restore(cache, snap)                          # outside the timer
    rows = []
    for T in ladder:
        best = min(samples[T])
        print(f"    {T:>10}{best:>12.1f}{best / T:>12.3f}")
        rows.append({"tokens": T, "ms_min": round(best, 3)})
    a, b, r2 = fit_line(ladder, [min(samples[T]) for T in ladder])
    print(f"    fit  t(T) = {a:.1f} ms + {b:.4f} ms/token   (R^2 = {r2:.4f})")
    print(f"    fixed floor is {a / (a + b * 16) * 100:.0f}% of a 16-token forward")
    del cache
    return {"cache_len": cache_len, "rows": rows,
            "fit_intercept_ms": round(a, 4), "fit_slope_ms_per_token": round(b, 6),
            "fit_r2": round(r2, 5)}


class _StubLayer:
    def __init__(self, n, n_kv, d, dtype):
        self.keys = torch.randn(1, n_kv, n, d, dtype=dtype)
        self.values = torch.randn(1, n_kv, n, d, dtype=dtype)
        self.true_length = n


class _StubCache:
    def __init__(self, n_layers, n, n_kv, d, dtype):
        self.layers = [_StubLayer(n, n_kv, d, dtype) for _ in range(n_layers)]


def part_a_restore(cfg, dtype, lengths, rounds, reps, q):
    """Cost of the cache copy `score_with_query` performs at every boundary.

    Shapes alone determine this, so it is timed on stub tensors -- which is also
    the only way to reach lengths this machine cannot prefill on CPU.

    THE CACHE MUST BE GROWN BY q FIRST, and the first version of this function
    was not. `_restore` truncates with `keys[:, :, :n].contiguous()`; when n is
    already the full length that slice is the whole tensor, `.contiguous()`
    returns `self`, and the measurement times a no-op. It read 0.15 ms for a
    939 MB copy -- 6 TB/s, which is the tell. Here the stub holds n + q
    positions and the snapshot says n, so the truncation is real, exactly as it
    is after the question has been appended.

    The per-rep re-pointing restores the grown tensors so the next rep truncates
    again; it is 2 * n_layers attribute assignments and is inside the timer.
    """
    n_kv = cfg.num_key_value_heads
    d = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n_layers = cfg.num_hidden_layers
    print(f"\n  A2. _restore: truncating {q} question rows off a cache, all "
          f"{n_layers} layers, {n_kv} KV heads, d={d}")
    print(f"    {'cache length':>14}{'ms (min)':>12}{'MB kept':>12}{'GB/s':>10}")
    rows = []
    for n in lengths:
        c = _StubCache(n_layers, n + q, n_kv, d, dtype)
        grown = [(l.keys, l.values) for l in c.layers]
        snap = [(n, n)] * n_layers
        samples = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            for _ in range(reps):
                for layer, (k, v) in zip(c.layers, grown):
                    layer.keys, layer.values = k, v
                _restore(c, snap)
            samples.append((time.perf_counter() - t0) / reps * 1000.0)
        mb = 2 * n_layers * n_kv * n * d * torch.finfo(dtype).bits / 8 / 1e6
        best = min(samples)
        # Read + write, hence 2 * mb: a copy touches both sides.
        gbps = 2 * mb / 1e3 / (best / 1e3) if best else float("nan")
        print(f"    {n:>14,}{best:>12.2f}{mb:>12.1f}{gbps:>10.1f}")
        rows.append({"cache_len": n, "q": q, "ms_min": round(best, 4),
                     "mb_kept": round(mb, 2), "effective_gb_s": round(gbps, 1)})
        del c, grown
    return rows


def part_a_record(model, dtype, cache_len, sizes, window, rounds):
    """Cost of the score accumulation itself, on the SAME forward, on and off.

    This is the prefill-side counterpart of the ~22% decode figure that
    measure_decode's evict/evict_noscore arms already give. It has to be an
    A-part measurement rather than a B-part one: in part B the blind and aware
    arms differ in recording AND in what they do at each boundary, so their
    chunk totals cannot isolate it, and the difference they show is inside
    round-to-round noise. Here nothing varies but the flag.
    """
    print(f"\n  A3. score accumulation on/off, same forward, cache {cache_len:,}")
    print(f"    {'tokens T':>10}{'recording off':>15}{'recording on':>14}"
          f"{'tax':>9}")
    cache = prefill_to(model, cache_len, dtype)
    snap = _snapshot(cache)
    rows = []
    samples = {(T, on): [] for T in sizes for on in (False, True)}
    for _ in range(rounds):
        for T in sizes:
            for on in (False, True):                       # interleaved
                ids = torch.randint(0, 30000, (1, T), device=DEV)
                mask = causal_lower_right(T, cache_len + T, DEV, dtype)
                pos = torch.arange(cache_len, cache_len + T,
                                   device=DEV).unsqueeze(0)
                sig.reset_scores()
                sig.set_recording(on, window=min(window, T))
                t0 = time.perf_counter()
                model(ids, past_key_values=cache, position_ids=pos,
                      attention_mask={"full_attention": mask},
                      use_cache=True, logits_to_keep=1)
                samples[(T, on)].append((time.perf_counter() - t0) * 1000.0)
                sig.set_recording(False)
                _restore(cache, snap)                      # outside the timer
    sig.reset_scores()
    for T in sizes:
        off, on = min(samples[(T, False)]), min(samples[(T, True)])
        print(f"    {T:>10}{off:>15.1f}{on:>14.1f}{on / off:>8.3f}x")
        rows.append({"tokens": T, "cache_len": cache_len,
                     "ms_off": round(off, 3), "ms_on": round(on, 3),
                     "tax": round(on / off, 4)})
    del cache
    return rows


# --------------------------------------------------------------------------- #
# B. end-to-end, self-instrumented
# --------------------------------------------------------------------------- #

class Tape:
    """Timestamps taken inside the prefill loop, so the split needs no arms.

    `compress_fn` is the only hook `chunked_prefill` offers, and it fires after
    every chunk but the last. The interval from the previous exit to this entry
    is that chunk's forward; the tail after the final exit is the last chunk's.
    """

    def __init__(self):
        self.chunk_ms, self.query_ms, self.compress_ms = [], [], []
        self.mark = None

    def start(self):
        self.mark = time.perf_counter()

    def boundary(self):
        now = time.perf_counter()
        self.chunk_ms.append((now - self.mark) * 1000.0)
        self.mark = now

    def phase(self, bucket):
        now = time.perf_counter()
        getattr(self, bucket).append((now - self.mark) * 1000.0)
        self.mark = now

    def finish(self):
        self.chunk_ms.append((time.perf_counter() - self.mark) * 1000.0)

    def total(self):
        return sum(self.chunk_ms) + sum(self.query_ms) + sum(self.compress_ms)


def run_arm(model, arm, ids, qids, chunk, policy, window, dtype):
    """One prefill under one configuration. Returns a Tape.

    blind  : recording ON over the chunks, ranked by each chunk's own last rows.
    aware  : recording OFF over the chunks, question run against the cache at
             every boundary and the pass undone.
    plain  : neither -- vanilla chunked prefill, the no-mechanism floor.
    """
    cache = sig.SignificanceCache(config=model.config, policy=None)
    sig.reset_scores()
    sig.set_recording(arm == "blind", window=window)
    tape = Tape()

    def compress_fn(c, ci):
        tape.boundary()
        if arm == "aware":
            score_with_query(model, c, qids,
                             position_base=c.layers[0].true_length, sig=sig)
            tape.phase("query_ms")
        sig.compress(c, policy)
        tape.phase("compress_ms")

    tape.start()
    chunked_prefill(model, ids, cache, chunk=chunk, logits_to_keep=1,
                    compress_fn=(None if arm == "plain" else compress_fn))
    tape.finish()
    sig.set_recording(False)
    kept = stored_length(cache)
    del cache
    return tape, kept


def part_b(model, dtype, configs, rounds, q_tokens, budget_frac, window):
    out = []
    arms = ("plain", "blind", "aware")
    for n_tok, chunk in configs:
        budget = max(64, int(chunk * budget_frac))
        ids = torch.randint(0, 30000, (1, n_tok), device=DEV)
        qids = torch.randint(0, 30000, (1, q_tokens), device=DEV)
        policy = sig.Policy(budget=budget, sinks=4, recent=32, pool=7,
                            pool_back=6, pool_fwd=0, normalize=True)
        best = {}
        for r in range(rounds):
            for arm in arms:                               # interleaved
                tape, kept = run_arm(model, arm, ids, qids, chunk, policy,
                                     window, dtype)
                cur = best.get(arm)
                if cur is None or tape.total() < cur[0].total():
                    best[arm] = (tape, kept)
            print(f"      round {r + 1}/{rounds} done, RSS {rss_mb():.0f} MB",
                  flush=True)
        nb = len(best["aware"][0].query_ms)
        print(f"\n  B. n={n_tok:,} chunk={chunk} budget={budget} "
              f"Q={q_tokens} -- {nb} boundaries")
        print(f"    {'arm':<8}{'total ms':>11}{'chunks':>10}{'query':>9}"
              f"{'compress':>10}{'kept':>8}")
        for arm in arms:
            t, kept = best[arm]
            print(f"    {arm:<8}{t.total():>11.0f}{sum(t.chunk_ms):>10.0f}"
                  f"{sum(t.query_ms):>9.0f}{sum(t.compress_ms):>10.0f}{kept:>8}")
        aw = best["aware"][0]
        bl = best["blind"][0]
        pl = best["plain"][0]
        q_share = sum(aw.query_ms) / aw.total()
        per_bnd = sum(aw.query_ms) / max(1, nb)
        per_chunk = sum(aw.chunk_ms) / len(aw.chunk_ms)
        naive = q_tokens / chunk
        print(f"\n    query-aware pass = {q_share * 100:.1f}% of aware prefill "
              f"({per_bnd:.0f} ms per boundary vs {per_chunk:.0f} ms per chunk)")
        print(f"    token-ratio prediction was {naive * 100:.2f}% -- "
              f"measured overstates it {per_bnd / per_chunk / naive:.0f}x")
        print(f"    aware / blind = {aw.total() / bl.total():.3f}x   "
              f"aware / plain = {aw.total() / pl.total():.3f}x   "
              f"blind / plain = {bl.total() / pl.total():.3f}x")
        out.append({
            "n_tokens": n_tok, "chunk": chunk, "budget": budget,
            "q_tokens": q_tokens, "boundaries": nb,
            "arms": {a: {"total_ms": round(best[a][0].total(), 2),
                         "chunk_ms": round(sum(best[a][0].chunk_ms), 2),
                         "query_ms": round(sum(best[a][0].query_ms), 2),
                         "compress_ms": round(sum(best[a][0].compress_ms), 2),
                         "kept": best[a][1]} for a in arms},
            "query_share_of_aware": round(q_share, 5),
            "ms_per_boundary": round(per_bnd, 2),
            "ms_per_chunk_aware": round(per_chunk, 2),
            "token_ratio_prediction": round(naive, 6),
            "overstatement_factor": round(per_bnd / per_chunk / naive, 2),
            "aware_over_blind": round(aw.total() / bl.total(), 4),
            "aware_over_plain": round(aw.total() / pl.total(), 4),
            "blind_over_plain": round(bl.total() / pl.total(), 4),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--rounds", type=int, default=3,
                    help="rounds for part B, which loads the model repeatedly")
    ap.add_argument("--rounds-a", type=int, default=5,
                    help="rounds for the cheap component measurements")
    ap.add_argument("--q-tokens", type=int, default=16)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--ladder", type=int, nargs="+",
                    default=[8, 16, 32, 64, 128, 256, 512])
    ap.add_argument("--ladder-cache", type=int, default=512)
    ap.add_argument("--restore-lengths", type=int, nargs="+",
                    default=[512, 2048, 8192, 32768])
    ap.add_argument("--configs", type=str, default="2048:512,4096:1024",
                    help="n_tokens:chunk pairs")
    ap.add_argument("--budget-frac", type=float, default=0.25)
    ap.add_argument(
        "--out",
        default="experiments/kv_significance/results/measure_query_prefill.json")
    ap.add_argument("--skip-b", action="store_true")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    from transformers import AutoModelForCausalLM
    print(f"  loading {args.model} on CPU as {args.dtype} "
          f"(RSS before {rss_mb():.0f} MB)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(ROOT / args.model),
                                                 dtype=dtype, device_map="cpu")
    model.eval()
    sig.patch_attention()
    print(f"  loaded, RSS {rss_mb():.0f} MB, threads {torch.get_num_threads()}",
          flush=True)

    a = part_a(model, dtype, args.ladder, args.ladder_cache, args.rounds_a)
    a["restore"] = part_a_restore(model.config, dtype, args.restore_lengths,
                                  args.rounds_a, reps=5, q=args.q_tokens)
    a["recording"] = part_a_record(model, dtype, args.ladder_cache,
                                   [args.q_tokens, args.ladder[-1]],
                                   args.window, args.rounds_a)
    b = []
    if not args.skip_b:
        configs = [tuple(int(x) for x in c.split(":"))
                   for c in args.configs.split(",")]
        b = part_b(model, dtype, configs, args.rounds, args.q_tokens,
                   args.budget_frac, args.window)

    # The GPU guarantee. Every tensor above is CPU, so no CUDA context was ever
    # created; this asserts it rather than asserting the env var, which this
    # torch build ignores.
    assert not torch.cuda.is_initialized(), "a CUDA context was created"

    p = ROOT / args.out
    p.write_text(json.dumps({
        "model": args.model, "device": "cpu", "dtype": args.dtype,
        "threads": torch.get_num_threads(), "rounds": args.rounds,
        "rounds_components": args.rounds_a,
        "statistic": "minimum across interleaved rounds",
        "cuda_initialized": torch.cuda.is_initialized(),
        "component_ladder": a, "end_to_end": b,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  wrote {p}")
    print("  CUDA context never created: "
          f"torch.cuda.is_initialized() = {torch.cuda.is_initialized()}")


if __name__ == "__main__":
    main()
