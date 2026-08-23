#!/usr/bin/env python3
"""Run gates C5/C6/C7 for a checkpoint, comparably with the prior Rineya artifacts.

Comparability is the whole point, so every detail that affects a score is taken
from the upstream reference implementation rather than reinvented:

  * record normalisation -- the trailing assistant turn is the gold `reference`
    and is removed from `messages`, so the prompt ends at the generation point
    (dataset.py in the reference lab)
  * prompt -- `add_generation_prompt=True, enable_thinking=False`
  * generation -- max_new_tokens 32, greedy, `eos_only` response control, which
    means no extra stopping criteria and no sentence trimming
  * reply -- `decode(new_tokens, skip_special_tokens=True).strip()`, then
    `finalize_response(...)` from the vendored response_control
  * scoring -- corpus profile built from the DATASET records (not results), and
    `visible_token_count` passed as *both* token_count and visible_model_tokens

verify_scorer.py confirms this pipeline reproduces upstream scores exactly on
stored replies, so a difference measured here is a difference in the model.

Results are appended per case, so progress is observable and a killed run resumes
by id instead of starting over.

Usage:
  python scripts/eval/run_gates.py --dataset chat_dev --out artifacts/gate_runs/r2_base_chat
  python scripts/eval/run_gates.py --dataset rag_dev  --adapter <lora_dir> --out ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

import memory as mem  # noqa: E402
from scripts.common.quality_evaluation import (  # noqa: E402
    QUALITY_EVALUATION_VERSION, aggregate_quality_rows, build_corpus_profile,
    evaluate_response, load_quality_config,
)
from scripts.common.response_control import (  # noqa: E402
    build_stopping_criteria, finalize_response, infer_stop_reason,
)

DATASETS = {"chat_dev": "artifacts/eval_sets/chat_dev.jsonl",
            "rag_dev": "artifacts/eval_sets/rag_dev.jsonl",
            "rag_test_SEALED": "artifacts/eval_sets/rag_test_SEALED.jsonl"}


def normalize(record: dict) -> dict:
    """Mirror the reference loader: gold comes out of messages into `reference`."""
    messages = [dict(m) for m in record.get("messages") or []]
    reference = record.get("reference")
    if messages and messages[-1].get("role") == "assistant":
        reference = str(messages[-1].get("content", ""))
        messages = messages[:-1]
    out = dict(record)
    out["messages"] = messages
    out["reference"] = reference
    out["category"] = str(record.get("category")
                          or (record.get("metadata") or {}).get("category")
                          or (record.get("metadata") or {}).get("reply_class")
                          or "uncategorized")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="artifacts/rineya-core-en32k-28L")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cap-mib", type=float, default=4096.0)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                    help="cpu lets a gate run share a machine whose GPU is busy")
    args = ap.parse_args()

    assets = json.loads((ROOT / "manifests" / "eval_assets.json").read_text(encoding="utf-8"))
    protocol = assets["generation_protocol"]
    ds_path = ROOT / DATASETS.get(args.dataset, args.dataset)
    out_dir = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    progress_path = out_dir / "progress.jsonl"

    if "SEALED" in ds_path.name:
        raise SystemExit("refusing to evaluate the sealed test set; it is reserved "
                         "for a frozen rung. Use rag_dev for gating.")

    records = [normalize(json.loads(l)) for l in
               ds_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        records = records[:args.limit]
    done = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[str(row["id"])] = row
        print(f"resuming: {len(done):,} of {len(records):,} cases already scored")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    cap = mem.apply_cap(args.cap_mib if use_cuda else None)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(ROOT / args.model)
    model = AutoModelForCausalLM.from_pretrained(ROOT / args.model, dtype=dtype)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.config.use_cache = True
    model = model.to(device).eval()

    print(f"dataset   : {ds_path.name}  ({len(records):,} cases)")
    print(f"model     : {args.model}" + (f"  + adapter {args.adapter}" if args.adapter else "  (no adapter)"))
    print(f"device    : {device}  cap {f'{cap:,.0f} MiB' if cap else 'n/a'}")
    print(f"protocol  : max_new_tokens={protocol['max_new_tokens']} "
          f"{protocol['response_control']} greedy={not protocol['do_sample']}")
    print(f"scorer    : v{QUALITY_EVALUATION_VERSION}\n")

    config = load_quality_config()
    corpus_profile = build_corpus_profile(records)
    pad_id = tokenizer.pad_token_id
    started = time.time()

    for index, record in enumerate(records):
        rid = str(record["id"])
        if rid in done:
            continue
        rendered = tokenizer.apply_chat_template(
            record["messages"], tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        inputs = tokenizer(rendered, return_tensors="pt").to(device)
        stopping, criterion = build_stopping_criteria(
            tokenizer, int(inputs.input_ids.shape[1]),
            str(protocol["response_control"]),
            int(protocol["min_new_tokens_before_boundary"]))
        kwargs = {"max_new_tokens": int(protocol["max_new_tokens"]),
                  "do_sample": bool(protocol["do_sample"]),
                  "pad_token_id": pad_id,
                  "eos_token_id": tokenizer.eos_token_id}
        if stopping is not None:
            kwargs["stopping_criteria"] = stopping
        with torch.inference_mode():
            generated = model.generate(**inputs, **kwargs)
        new_tokens = generated[0, inputs.input_ids.shape[1]:]

        raw_with_special = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
        ordinary = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        finalized = finalize_response(ordinary, str(protocol["response_control"]))
        reply = finalized.text
        visible = len(tokenizer(reply, add_special_tokens=False)["input_ids"]) if reply else 0
        stop_reason = infer_stop_reason(
            new_tokens.tolist(), eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=int(protocol["max_new_tokens"]), criterion=criterion)

        quality = evaluate_response(record, reply, visible,
                                    visible_model_tokens=visible,
                                    config=config, corpus_profile=corpus_profile)
        row = {"id": rid, "category": record["category"], "reply": reply,
               "raw_reply": raw_with_special, "reference": record.get("reference"),
               "generated_tokens": int(new_tokens.numel()),
               "visible_generated_tokens": visible,
               "prompt_tokens": int(inputs.input_ids.shape[1]),
               "stop_reason": stop_reason,
               "response_control_trimmed": bool(finalized.trimmed),
               **quality}
        with results_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        done[rid] = row

        n = len(done)
        if n % 10 == 0 or n == len(records):
            per = (time.time() - started) / max(n - (len(done) - index - 1), 1)
            with progress_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps({
                    "done": n, "total": len(records),
                    "elapsed_s": round(time.time() - started, 1),
                    **mem.snapshot()}) + "\n")
            if n % 50 == 0 or n == len(records):
                print(f"  {n:>4,d}/{len(records):,}  "
                      f"{100*n/len(records):5.1f}%  "
                      f"{time.time()-started:6.0f}s elapsed")

    rows = [done[str(r["id"])] for r in records if str(r["id"]) in done]
    summary = aggregate_quality_rows(rows)
    summary["quality_evaluation_version"] = QUALITY_EVALUATION_VERSION
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    per_cat: dict[str, list[float]] = {}
    for row in rows:
        per_cat.setdefault(row["category"], []).append(float(row.get("quality_score", 0.0)))
    (out_dir / "per_category.json").write_text(json.dumps(
        {k: {"n": len(v), "mean_quality_score": sum(v)/len(v)}
         for k, v in sorted(per_cat.items())}, indent=2), encoding="utf-8")

    (out_dir / "evaluation_manifest.json").write_text(json.dumps({
        "dataset": str(ds_path.relative_to(ROOT)).replace("\\", "/"),
        "dataset_sha256": assets["eval_sets"].get(
            {"chat_dev": "chat_dev_510", "rag_dev": "rag_dev_752"}.get(args.dataset, ""), {}
        ).get("sha256"),
        "model": args.model, "adapter": args.adapter,
        "case_count": len(rows), "protocol": protocol,
        "scorer_invocation": assets["scorer_invocation"],
        "quality_evaluation_version": QUALITY_EVALUATION_VERSION,
        "elapsed_s": round(time.time() - started, 1),
    }, indent=2), encoding="utf-8")

    print(f"\n--- {args.dataset} : {len(rows):,} cases in {(time.time()-started)/60:.1f} min ---")
    for key in ("mean_quality_score", "quality_pass_rate", "intelligibility_pass_rate",
                "critical_defect_case_rate", "contract_violation_case_rate",
                "mean_contract_fulfillment_score"):
        if key in summary:
            print(f"  {key:34s} {float(summary[key]):.4f}")

    print("\n  vs historical anchors (same scorer, same cases):")
    for name, payload in assets.get("anchors", {}).items():
        if not name.startswith(args.dataset.split("_")[0]):
            continue
        s = payload.get("summary", {})
        if "mean_quality_score" in s:
            delta = float(summary["mean_quality_score"]) - float(s["mean_quality_score"])
            print(f"    {name:34s} {float(s['mean_quality_score']):.4f}   "
                  f"this run {delta:+.4f}")
    print(f"\nwrote {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
