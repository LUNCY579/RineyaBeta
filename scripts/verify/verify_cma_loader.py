#!/usr/bin/env python3
"""Load, merge, and forward-test exact and rank-matched CMA adapters."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="artifacts/rineya-core-en32k-28L")
    parser.add_argument("--exact", default="artifacts/adapters/rge_m5_cma_exact_r96")
    parser.add_argument("--rank", default="artifacts/adapters/rge_m5_cma_rank16")
    parser.add_argument("--evidence",
                        default="experiments/rge_peft/evidence/m5_cma_loader_gpu.json")
    args = parser.parse_args()
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(4096 * (1 << 20), total) / total)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(ROOT / args.base)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with one word: ready?"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tokens = tokenizer(rendered, return_tensors="pt").to(device)
    results = []
    for label, relative, expected_rank in (
            ("exact", args.exact, 96), ("rank_matched", args.rank, 16)):
        adapter = ROOT / relative
        config = PeftConfig.from_pretrained(adapter)
        if int(config.r) != expected_rank:
            raise AssertionError(f"{label} adapter rank mismatch")
        model = AutoModelForCausalLM.from_pretrained(ROOT / args.base, dtype=dtype)
        model = PeftModel.from_pretrained(model, adapter).to(device).eval()
        with torch.inference_mode():
            logits = model(**tokens).logits[:, -1, :].float()
        if not bool(torch.isfinite(logits).all()):
            raise AssertionError(f"{label} adapter produced non-finite logits")
        merged = model.merge_and_unload().eval()
        with torch.inference_mode():
            merged_logits = merged(**tokens).logits[:, -1, :].float()
        maximum_merge_delta = float((logits - merged_logits).abs().max().item())
        results.append({
            "label": label,
            "adapter": relative,
            "rank": expected_rank,
            "alpha": int(config.lora_alpha),
            "finite_forward": True,
            "merge_and_unload": True,
            "max_logit_delta_peft_vs_merged": maximum_merge_delta,
        })
        del model, merged, logits, merged_logits
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {
        "schema_version": 1,
        "status": "PASS",
        "device": str(device),
        "adapters": results,
        "sealed_data_accessed": False,
    }
    evidence = ROOT / args.evidence
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
