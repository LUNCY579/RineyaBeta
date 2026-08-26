#!/usr/bin/env python3
"""Quality sweep for significance-gated eviction, protocol-identical to run_gates.py.

Same protocol as the gate baselines and the attention-shape sweep: chat template
with `enable_thinking=False`, `max_new_tokens=32`, greedy, `eos_only`, visible
token count passed to the scorer as both `token_count` and `visible_model_tokens`.

`model.generate` is used unmodified -- eviction happens inside the cache, so this
cannot drift from the reference the way a hand-rolled decode loop would.

Budget can be absolute (`--budget 128`) or a fraction of each prompt's length
(`--keep-frac 0.5`). The fraction form is usually the meaningful one, because
these prompts are 41-330 tokens and a fixed budget means very different
compression ratios across the set.

Run `--budget 0` first: eviction off, which produces the reference every other
run is compared against.

Usage:
    python src/run_eval_sig.py --budget 0 --dataset chat_dev
    python src/run_eval_sig.py --keep-frac 0.5 --dataset chat_dev --compare-to off
    python src/run_eval_sig.py --keep-frac 0.25 --pool 7 --dataset rag_dev --compare-to off
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments/attention_shape/src"))
sys.path.insert(0, str(REPO))

from scripts.kv import significance as sig  # noqa: E402
from compat import force_repeat_kv  # noqa: E402

MODEL = "artifacts/rineya-core-en32k-28L"
DATASETS = {
    "chat_dev": "artifacts/eval_sets/chat_dev.jsonl",
    "rag_dev": "artifacts/eval_sets/rag_dev.jsonl",
}


def normalize(record: dict) -> dict:
    rec = dict(record)
    msgs = list(rec.get("messages") or [])
    reference = None
    if msgs and msgs[-1].get("role") == "assistant":
        reference = str(msgs[-1].get("content") or "")
        msgs = msgs[:-1]
    rec["messages"] = msgs
    rec["reference"] = rec.get("reference") or reference
    meta = rec.get("metadata") or {}
    rec["category"] = (rec.get("category") or meta.get("category")
                       or meta.get("reply_class") or "uncategorized")
    return rec


def load_records(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(normalize(json.loads(line)))
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="chat_dev", choices=sorted(DATASETS))
    ap.add_argument("--budget", type=int, default=0, help="0 = eviction off")
    ap.add_argument("--keep-frac", type=float, default=None,
                    help="budget as a fraction of each prompt's length")
    ap.add_argument("--sinks", type=int, default=4)
    ap.add_argument("--recent", type=int, default=32)
    ap.add_argument("--recent-frac", type=float, default=None,
                    help="recent window as a fraction of the budget. With a fixed "
                         "recent=32 and a small budget the always-kept regions "
                         "consume nearly the whole allowance -- at keep 0.25 on a "
                         "150-token prompt only 5%% of the budget was actually "
                         "significance-selected, making that cell StreamingLLM "
                         "rather than a test of scoring.")
    ap.add_argument("--pool", type=int, default=1)
    # Cornerstone 2 -- directional pooling. These existed in passkey.py from the
    # day the mechanism was found and were never added here, so every
    # natural-language number in this programme was measured with the CENTRED
    # window that cornerstone 2 superseded. Without these flags the mechanism
    # cannot be evaluated on language at all.
    ap.add_argument("--pool-back", type=int, default=None,
                    help="backward reach of the pooling window. Cornerstone 2's "
                         "result is that reach BACKWARD is what decides; forward "
                         "reach is inert. Requires --pool > 1.")
    ap.add_argument("--pool-fwd", type=int, default=None,
                    help="forward reach. Set 0 with --pool-back for the causal "
                         "window. Requires --pool > 1.")
    ap.add_argument("--grow", type=float, default=None,
                    help="derived reach: geodesic dilation with gate "
                         "s > median(s)*mult. Beats a single fixed reach held "
                         "constant across span lengths.")
    ap.add_argument("--grow-max", type=int, default=64,
                    help="dilation cap; only meaningful with --grow.")
    ap.add_argument("--min-heavy-frac", type=float, default=0.0,
                    help="B1: reserve at least this FRACTION of the budget for "
                         "score-chosen slots. The recent window yields to it. "
                         "Prevents the degenerate case where sinks+recent consume "
                         "the whole budget and the policy is silently StreamingLLM "
                         "-- which fired 17,360 times on rag_dev at keep 0.25.")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--no-sinks", action="store_true", help="disable sink protection")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--compare-to", default=None)
    args = ap.parse_args()

    force_repeat_kv(verbose=False)
    sig.patch_attention()
    sig.reset_heavy_zero_count()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scripts.common.quality_evaluation import (  # noqa: E402
        aggregate_quality_rows, build_corpus_profile, evaluate_response,
        load_quality_config)

    path = REPO / MODEL
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
    ).to(args.device).eval()

    protocol = json.loads(
        (REPO / "manifests/eval_assets.json").read_text(encoding="utf-8")
    ).get("generation_protocol", {})
    max_new = protocol.get("max_new_tokens", 32)

    records = load_records(REPO / DATASETS[args.dataset], args.limit)
    evicting = bool(args.budget) or args.keep_frac is not None
    tag = args.tag or ("off" if not evicting else
                       (f"frac{args.keep_frac}" if args.keep_frac is not None
                        else f"b{args.budget}")
                       + (f"-pool{args.pool}" if args.pool > 1 else "")
                       + (f"-rf{args.recent_frac}" if args.recent_frac is not None else "")
                       + ("-nosink" if args.no_sinks else ""))

    print(f"variant {tag}   dataset {args.dataset} ({len(records)} cases)   "
          f"device {args.device}")
    print(f"eviction {'ON' if evicting else 'OFF'}"
          + (f"   budget {'frac ' + str(args.keep_frac) if args.keep_frac is not None else args.budget}"
             f"   sinks {0 if args.no_sinks else args.sinks}   recent {args.recent}"
             f"   pool {args.pool}"
             + (f"   window ({args.pool_back},{args.pool_fwd}) CAUSAL"
                if args.pool_fwd == 0 and args.pool_back
                else f"   window ({args.pool_back},{args.pool_fwd})"
                if args.pool_back is not None else "   window CENTRED")
             + (f"   grow {args.grow}" if args.grow else "")
             + f"   attn-window {args.window}" if evicting else ""))

    config = load_quality_config()
    corpus_profile = build_corpus_profile(records)

    rows, t0 = [], time.time()
    kept_ratios = []
    recents_used = []
    windows_used = []
    for i, rec in enumerate(records):
        prompt = tok.apply_chat_template(rec["messages"], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        enc = tok(prompt, return_tensors="pt").to(args.device)
        n_prompt = enc["input_ids"].shape[-1]

        policy = None
        if evicting:
            budget = (max(1, int(round(args.keep_frac * n_prompt)))
                      if args.keep_frac is not None else args.budget)
            recent = (max(1, int(round(args.recent_frac * budget)))
                      if args.recent_frac is not None else args.recent)
            policy = sig.Policy(budget=budget, sinks=args.sinks, recent=recent,
                                protect_sinks=not args.no_sinks, pool=args.pool,
                                pool_back=args.pool_back, pool_fwd=args.pool_fwd,
                                grow=args.grow, grow_max=args.grow_max,
                                min_heavy_frac=args.min_heavy_frac)
            # Record what the policy ACTUALLY used. Writing args.recent here --
            # which is what this file used to do -- makes every --recent-frac run
            # look like a fixed recent=32 run in the manifest, and summarize.py
            # then reported the significance-selected share as 13%/37% when the
            # truth is 63.7%/67.4%. The one column whose purpose is to expose
            # StreamingLLM-in-disguise was reporting the fix for it as a no-op.
            recents_used.append(policy.n_recent())
            windows_used.append(tuple(policy.pool_window()))

        sig.reset_scores()
        sig.set_recording(evicting, window=args.window)
        cache = sig.SignificanceCache(config=model.config, policy=policy)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id,
                                 past_key_values=cache)
        sig.set_recording(False)
        stored = cache.stored_lengths()[0]
        kept_ratios.append(stored / max(n_prompt, 1))

        new_tokens = out[0, n_prompt:]
        reply = tok.decode(new_tokens, skip_special_tokens=True).strip()
        visible = len(tok(reply, add_special_tokens=False)["input_ids"])
        quality = evaluate_response(rec, reply, visible, visible_model_tokens=visible,
                                    config=config, corpus_profile=corpus_profile)
        rows.append({"id": rec.get("id"), "category": rec["category"], "reply": reply,
                     "prompt_tokens": n_prompt, "stored_kv": stored,
                     "visible_generated_tokens": visible, **quality})
        if (i + 1) % 100 == 0:
            # Memory alongside progress. A run of this harness slowed monotonically
            # (233/245/571/528 s per 100 cases) and then died with a CUDA illegal
            # memory access; wall-clock alone could not distinguish a leak from
            # thermal throttling, and there was no allocator number to look at.
            # `reserved` is what matters for fragmentation: allocated can stay
            # flat while reserved climbs, which is the signature that precedes an
            # allocator failure.
            if torch.cuda.is_available():
                al = torch.cuda.memory_allocated() / 2 ** 20
                rs = torch.cuda.memory_reserved() / 2 ** 20
                print(f"  {i + 1}/{len(records)}  {time.time() - t0:.0f}s  "
                      f"alloc {al:.0f} MiB  reserved {rs:.0f} MiB", flush=True)
            else:
                print(f"  {i + 1}/{len(records)}  {time.time() - t0:.0f}s", flush=True)

    summary = aggregate_quality_rows(rows)
    elapsed = time.time() - t0
    mean_keep = sum(kept_ratios) / len(kept_ratios)
    print(f"\nmean_quality_score      {summary['mean_quality_score']:.5f}")
    print(f"quality_pass_rate       {summary['quality_pass_rate']:.5f}")
    print(f"intelligibility         {summary['intelligibility_pass_rate']:.5f}")
    print(f"mean kept / prompt      {mean_keep:.3f}")
    print(f"elapsed                 {elapsed:.1f}s")

    out_dir = EXP / "results" / f"{tag}__{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                          encoding="utf-8")
    per_cat: dict[str, dict] = {}
    for r in rows:
        b = per_cat.setdefault(r["category"], {"n": 0, "total": 0.0})
        b["n"] += 1
        b["total"] += float(r.get("quality_score") or 0.0)
    (out_dir / "per_category.json").write_text(json.dumps(
        {k: {"n": v["n"], "mean_quality_score": v["total"] / v["n"]}
         for k, v in sorted(per_cat.items())}, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "run_manifest.json").write_text(json.dumps({
        "tag": tag, "dataset": args.dataset, "eviction": evicting,
        "budget": args.budget, "keep_frac": args.keep_frac,
        "sinks": 0 if args.no_sinks else args.sinks,
        "recent_flag": args.recent,
        "recent": (round(sum(recents_used) / len(recents_used), 2)
                   if recents_used else args.recent),
        "recent_effective_mean": (round(sum(recents_used) / len(recents_used), 2)
                                  if recents_used else None),
        "recent_frac": args.recent_frac,
        "heavy_zero_count": sig.heavy_zero_count(),
        "pool": args.pool, "window": args.window,
        # Record the EFFECTIVE window, not the flags. Policy.pool_window()
        # defaults to (pool//2, pool//2) when --pool-back is unset, so a run
        # that recorded only the flags would write pool_back=null for a centred
        # window AND for a directional one that happened to be defaulted -- the
        # same ambiguity that made a heavy=0 run record as heavy=8. Any pin
        # asserting `pool_back` must be able to tell a cornerstone-2 run from a
        # cornerstone-1 run from the manifest alone.
        "pool_back": (windows_used[0][0] if windows_used else None),
        "pool_fwd": (windows_used[0][1] if windows_used else None),
        "pool_back_flag": args.pool_back, "pool_fwd_flag": args.pool_fwd,
        "pool_window_uniform": (len(set(windows_used)) <= 1) if windows_used else None,
        "grow": args.grow,
        "min_heavy_frac": args.min_heavy_frac,
        "grow_max": (args.grow_max if args.grow else None),
        "directional": bool(windows_used and windows_used[0][1] == 0
                            and windows_used[0][0] > 0),
        "mean_kept_ratio": round(mean_keep, 4),
        "case_count": len(rows), "protocol": protocol,
        "elapsed_s": round(elapsed, 1),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_dir}")

    if args.compare_to:
        ref_path = EXP / "results" / f"{args.compare_to}__{args.dataset}" / "results.jsonl"
        if not ref_path.exists():
            raise SystemExit(f"no reference run at {ref_path}")
        ref = {json.loads(l)["id"]: json.loads(l)
               for l in ref_path.read_text(encoding="utf-8").splitlines() if l.strip()}
        common = [r for r in rows if r["id"] in ref]
        same = sum(1 for r in common if r["reply"] == ref[r["id"]]["reply"])
        ref_mean = sum(float(ref[r["id"]].get("quality_score") or 0.0)
                       for r in common) / max(len(common), 1)
        print(f"\nvs {args.compare_to}: {same}/{len(common)} byte-identical "
              f"({100 * same / max(len(common), 1):.1f}%)")
        print(f"  quality {ref_mean:.5f} -> {summary['mean_quality_score']:.5f}  "
              f"({summary['mean_quality_score'] - ref_mean:+.5f})")


if __name__ == "__main__":
    main()
