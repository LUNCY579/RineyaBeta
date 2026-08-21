#!/usr/bin/env python3
"""Quality sweep over attention-shape variants, protocol-identical to `run_gates.py`.

This does not import `run_gates.py` (it hardcodes its own model loading and has
no hook for config overrides), but it reproduces its protocol exactly and reuses
the *same vendored scorer*, so numbers land on the same scale as the existing
baselines in `artifacts/gate_runs/`:

    r2_base_chat_dev   mean_quality_score 0.83360   (510 cases)
    r2_base_rag_dev    mean_quality_score 0.66818   (752 cases)

Protocol, matched to `run_gates.py:138-167`: chat template with
`enable_thinking=False` and `add_generation_prompt=True`; `max_new_tokens=32`;
greedy; `response_control="eos_only"`; reply decoded with
`skip_special_tokens=True` and stripped; visible token count passed to the
scorer as both `token_count` and `visible_model_tokens`.

**Start with `--self-check`.** It runs the unmodified base in `--no-compat` mode
and asserts the result reproduces the stored baseline exactly. Comparability is
proved before anything is compared (P4); a sweep whose harness silently differs
from the one that produced the baselines measures the harness, not the variants.

Note that `--self-check` and the variant sweep deliberately run in *different*
kernel modes, for the reason documented at the top of `main()`. Do not "fix" the
mismatch by patching the baseline run too -- that would silently re-introduce a
kernel confound into every comparison.

The headline test is `--compare-to`, which diffs replies case by case. For a
window wider than every prompt, sliding attention is mathematically a no-op, so
the correct result is **510/510 byte-identical**. Anything else is a bug in the
mask or cache path, not a finding about sliding windows.

Usage:
    python src/run_eval.py --self-check --dataset chat_dev
    python src/run_eval.py --variant swa-w512-g4 --dataset chat_dev --compare-to base-8KV
    python src/run_eval.py --variant swa-w128-g4 --dataset rag_dev
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from compat import force_repeat_kv  # noqa: E402
from variants import catalogue, load_variant  # noqa: E402

DATASETS = {
    "chat_dev": "artifacts/eval_sets/chat_dev.jsonl",
    "rag_dev": "artifacts/eval_sets/rag_dev.jsonl",
}
BASELINE_RUNS = {
    "chat_dev": "artifacts/gate_runs/r2_base_chat_dev",
    "rag_dev": "artifacts/gate_runs/r2_base_rag_dev",
}
SURGERY_ROOT = "artifacts/attn_shape"


def normalize(record: dict) -> dict:
    """Mirror of run_gates.py:57-71."""
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
            line = line.strip()
            if line:
                rows.append(normalize(json.loads(line)))
            if limit and len(rows) >= limit:
                break
    return rows


@torch.no_grad()
def generate(model, tokenizer, record: dict, protocol: dict, device: str) -> dict:
    prompt = tokenizer.apply_chat_template(
        record["messages"], tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    n_prompt = enc["input_ids"].shape[-1]
    out = model.generate(
        **enc,
        max_new_tokens=protocol.get("max_new_tokens", 32),
        do_sample=bool(protocol.get("do_sample", False)),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[0, n_prompt:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    visible = len(tokenizer(reply, add_special_tokens=False)["input_ids"])
    return {"reply": reply, "visible": visible, "prompt_tokens": n_prompt,
            "generated_tokens": int(new_tokens.shape[0])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="base-8KV")
    ap.add_argument("--dataset", default="chat_dev", choices=sorted(DATASETS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--compare-to", default=None,
                    help="name of a previous run in results/ to diff replies against")
    ap.add_argument("--no-compat", action="store_true",
                    help="leave enable_gqa alone (matches how run_gates.py ran)")
    ap.add_argument("--self-check", action="store_true",
                    help="run base-8KV and assert it reproduces the stored baseline")
    args = ap.parse_args()

    # Two modes, and the distinction is load-bearing.
    #
    #   --no-compat  leaves enable_gqa alone, exactly as run_gates.py ran. This is
    #                the ONLY mode comparable to the stored gate baselines, so
    #                --self-check forces it.
    #   default      forces repeat_kv. Required for the variant sweep, because
    #                sliding-window variants pass an explicit attention mask and
    #                therefore take the repeat_kv path whether we ask or not.
    #                Patching the baseline too is what makes the comparison a
    #                comparison of attention shapes rather than of SDPA kernels.
    #
    # The two are mathematically equivalent and numerically are not: measured on
    # chat_dev, switching kernels changes 23% of greedy replies while moving mean
    # quality by +0.00138. Aggregates hide this completely (P8) -- which is why
    # the reference for --compare-to is our own patched base run, never the
    # stored baseline.
    if args.self_check:
        args.no_compat = True
    if not args.no_compat:
        force_repeat_kv()

    from scripts.common.quality_evaluation import (  # noqa: E402
        aggregate_quality_rows, build_corpus_profile, evaluate_response,
        load_quality_config)

    variant_name = "base-8KV" if args.self_check else args.variant
    cat = {v.name: v for v in catalogue()}
    if variant_name not in cat:
        raise SystemExit(f"unknown variant {variant_name!r}; have {sorted(cat)}")
    variant = cat[variant_name]

    protocol = json.loads(
        (REPO / "manifests/eval_assets.json").read_text(encoding="utf-8")
    ).get("generation_protocol", {})

    records = load_records(REPO / DATASETS[args.dataset], args.limit)
    print(f"variant {variant_name}   dataset {args.dataset} ({len(records)} cases)   "
          f"device {args.device}")
    print(f"protocol max_new_tokens={protocol.get('max_new_tokens', 32)} "
          f"do_sample={protocol.get('do_sample', False)}")

    surgery_dir = (f"{SURGERY_ROOT}/r2-gqa{variant.surgery_kv_heads}kv"
                   if variant.needs_surgery else None)
    from transformers import AutoTokenizer
    tok_path = REPO / (surgery_dir or "artifacts/rineya-core-en32k-28L")
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    model = load_variant(variant, device=args.device, surgery_dir=surgery_dir)

    config = load_quality_config()
    corpus_profile = build_corpus_profile(records)

    rows, t0 = [], time.time()
    for i, rec in enumerate(records):
        gen = generate(model, tokenizer, rec, protocol, args.device)
        quality = evaluate_response(rec, gen["reply"], gen["visible"],
                                    visible_model_tokens=gen["visible"],
                                    config=config, corpus_profile=corpus_profile)
        rows.append({"id": rec.get("id"), "category": rec["category"],
                     "reply": gen["reply"], "prompt_tokens": gen["prompt_tokens"],
                     "visible_generated_tokens": gen["visible"], **quality})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(records)}  {time.time() - t0:.0f}s", flush=True)

    summary = aggregate_quality_rows(rows)
    elapsed = time.time() - t0
    print(f"\nmean_quality_score      {summary['mean_quality_score']:.5f}")
    print(f"quality_pass_rate       {summary['quality_pass_rate']:.5f}")
    print(f"intelligibility         {summary['intelligibility_pass_rate']:.5f}")
    print(f"critical_defect_rate    {summary['critical_defect_case_rate']:.5f}")
    print(f"elapsed                 {elapsed:.1f}s")

    out_dir = args.out or (HERE.parents[0] / "results" / f"{variant_name}__{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                          encoding="utf-8")
    per_cat: dict[str, dict] = {}
    for r in rows:
        b = per_cat.setdefault(r["category"], {"n": 0, "total": 0.0})
        b["n"] += 1
        b["total"] += float(r.get("quality_score") or 0.0)
    per_cat = {k: {"n": v["n"], "mean_quality_score": v["total"] / v["n"]}
               for k, v in sorted(per_cat.items())}
    (out_dir / "per_category.json").write_text(json.dumps(per_cat, indent=2) + "\n",
                                               encoding="utf-8")
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "run_manifest.json").write_text(json.dumps({
        "variant": variant_name, "dataset": args.dataset,
        "config_overrides": variant.config_overrides,
        "needs_surgery": variant.needs_surgery, "unhealed": variant.needs_surgery,
        "case_count": len(rows), "protocol": protocol, "elapsed_s": round(elapsed, 1),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_dir}")

    # ---- self-check against the stored baseline ---------------------------
    if args.self_check:
        ref = json.loads((REPO / BASELINE_RUNS[args.dataset] / "summary.json")
                         .read_text(encoding="utf-8"))
        d = abs(summary["mean_quality_score"] - ref["mean_quality_score"])
        print(f"\nself-check vs {BASELINE_RUNS[args.dataset]}")
        print(f"  stored   {ref['mean_quality_score']:.5f}")
        print(f"  measured {summary['mean_quality_score']:.5f}   |delta| {d:.2e}")
        if args.limit:
            print("  (partial run -- not a valid check; drop --limit)")
        elif d < 1e-9:
            print("  PASS - harness reproduces the baseline exactly")
        else:
            print("  FAIL - this harness does not match the one that made the "
                  "baselines; fix before trusting any comparison")
            raise SystemExit(1)

    # ---- reply-level diff -------------------------------------------------
    if args.compare_to:
        ref_path = HERE.parents[0] / "results" / f"{args.compare_to}__{args.dataset}" / "results.jsonl"
        if not ref_path.exists():
            raise SystemExit(f"no reference run at {ref_path}")
        ref_rows = {json.loads(l)["id"]: json.loads(l)
                    for l in ref_path.read_text(encoding="utf-8").splitlines() if l.strip()}
        same = sum(1 for r in rows
                   if r["id"] in ref_rows and r["reply"] == ref_rows[r["id"]]["reply"])
        n = sum(1 for r in rows if r["id"] in ref_rows)
        ref_mean = sum(float(ref_rows[r["id"]].get("quality_score") or 0.0)
                       for r in rows if r["id"] in ref_rows) / max(n, 1)
        print(f"\nreply diff vs {args.compare_to}: {same}/{n} byte-identical "
              f"({100 * same / max(n, 1):.1f}%)")
        print(f"  quality {ref_mean:.5f} -> {summary['mean_quality_score']:.5f}  "
              f"({summary['mean_quality_score'] - ref_mean:+.5f})")
        if same == n:
            print("  identical everywhere -- this variant is provably free at this "
                  "context length")


if __name__ == "__main__":
    main()
