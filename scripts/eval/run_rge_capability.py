#!/usr/bin/env python3
"""Run the frozen RGE targeted-capability validation protocol.

The runner is deliberately validation-only, resumable by case ID, and strict
about the frozen dataset hash.  Test, challenge, and sealed paths are refused.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "train"))
import memory as mem  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_fingerprint(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def reject_nonvalidation_path(path: Path) -> None:
    lowered = path.as_posix().casefold()
    forbidden = ("sealed", "/test", "test.", "/challenge", "challenge.")
    if any(token in lowered for token in forbidden):
        raise ValueError(f"refusing non-validation capability asset: {path}")
    if "validation" not in path.name.casefold():
        raise ValueError(f"capability asset must be explicitly named validation: {path}")


def extract_final(text: str) -> str:
    """Mirror Rineya-Puzzle-Reasoning-51k/scripts/evaluate_predictions.py."""
    text = text.strip()
    match = re.search(r"<final>(.*?)</final>", text, flags=re.I | re.S)
    if match:
        return match.group(1).strip()
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    if re.search(r"\bFinal\s*:", text, flags=re.I):
        return re.split(r"\bFinal\s*:\s*", text, flags=re.I)[-1].strip()
    return text


def normalize_answer(text: str) -> str:
    """Mirror the frozen dataset's case/space/punctuation normalizer."""
    text = text.casefold().strip().replace("→", "->")
    text = re.sub(r"[^\w\s\-\>]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def score_prediction(row: dict[str, Any], prediction: str) -> dict[str, Any]:
    extracted = extract_final(prediction)
    normalized = normalize_answer(extracted)
    accepted = [normalize_answer(str(answer)) for answer in row["accepted_answers"]]
    return {
        "extracted_prediction": extracted,
        "normalized_prediction": normalized,
        "correct": normalized in accepted,
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    ids = [str(row["id"]) for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("frozen capability asset contains duplicate IDs")
    required = {"id", "family", "difficulty", "prompt", "accepted_answers"}
    for row in records:
        missing = required - set(row)
        if missing:
            raise ValueError(f"capability row {row.get('id')} misses {sorted(missing)}")
    return records


def summarize(records: list[dict[str, Any]], done: dict[str, dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[tuple[str, str], list[int]] = collections.defaultdict(lambda: [0, 0])
    correct = 0
    words = []
    for record in records:
        row = done[str(record["id"])]
        value = int(bool(row["correct"]))
        correct += value
        words.append(len(str(row["extracted_prediction"]).split()))
        for kind in ("family", "difficulty", "answer_type"):
            name = str(record.get(kind, "unknown"))
            buckets[(kind, name)][0] += value
            buckets[(kind, name)][1] += 1

    def bucket(kind: str) -> dict[str, dict[str, float | int]]:
        return {
            name: {"correct": values[0], "total": values[1],
                   "accuracy": values[0] / values[1] if values[1] else 0.0}
            for (bucket_kind, name), values in sorted(buckets.items())
            if bucket_kind == kind
        }

    return {
        "schema_version": 1,
        "accuracy": correct / len(records) if records else 0.0,
        "correct": correct,
        "total": len(records),
        "coverage": len(done) / len(records) if records else 0.0,
        "mean_extracted_output_words": sum(words) / len(words) if words else 0.0,
        "by_family": bucket("family"),
        "by_difficulty": bucket("difficulty"),
        "by_answer_type": bucket("answer_type"),
    }


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/rge_peft/m6_capability_dev.json")
    parser.add_argument("--model", default="artifacts/rineya-core-en32k-28L")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cap-mib", type=float, default=5120.0)
    parser.add_argument("--limit", type=int, default=None,
                        help="diagnostic prefix only; never eligible as full-bank evidence")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    manifest_path = (ROOT / args.manifest).resolve()
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_path = (ROOT / str(frozen["asset"])).resolve()
    reject_nonvalidation_path(dataset_path)
    observed_dataset_sha = sha256_file(dataset_path)
    if observed_dataset_sha != frozen["sha256"]:
        raise SystemExit("frozen capability asset SHA256 mismatch")
    all_records = load_records(dataset_path)
    if len(all_records) != int(frozen["rows"]):
        raise SystemExit("frozen capability row count mismatch")
    observed_families = dict(sorted(collections.Counter(
        str(row["family"]) for row in all_records).items()))
    if observed_families != frozen["families"]:
        raise SystemExit("frozen capability family counts mismatch")
    records = all_records[:args.limit] if args.limit is not None else all_records

    model_path = (ROOT / args.model).resolve()
    adapter_path = (ROOT / args.adapter).resolve()
    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    progress_path = out_dir / "progress.jsonl"
    evaluation_manifest_path = out_dir / "evaluation_manifest.json"
    protocol = {
        **frozen["generation"],
        "batch_size": args.batch_size,
        "diagnostic_limit": args.limit,
    }
    fingerprints = {
        "dataset_sha256": observed_dataset_sha,
        "manifest_sha256": sha256_file(manifest_path),
        "model_fingerprint": tree_fingerprint(model_path),
        "adapter_fingerprint": tree_fingerprint(adapter_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }

    done: dict[str, dict[str, Any]] = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rid = str(row["id"])
            if rid in done:
                raise SystemExit(f"duplicate resumed prediction ID: {rid}")
            done[rid] = row
        expected = {str(record["id"]) for record in records}
        if not set(done).issubset(expected):
            raise SystemExit("existing results contain IDs outside this run")
        if not evaluation_manifest_path.is_file():
            raise SystemExit("refusing results resume without evaluation_manifest.json")
        prior = json.loads(evaluation_manifest_path.read_text(encoding="utf-8"))
        for key, value in fingerprints.items():
            if prior.get(key) != value:
                raise SystemExit(f"resume fingerprint mismatch: {key}")
        if prior.get("protocol") != protocol:
            raise SystemExit("resume generation protocol mismatch")
        print(f"resuming: {len(done):,} of {len(records):,} cases already complete")

    run_manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "IN_PROGRESS",
        "role": "targeted_capability_dev",
        "sealed_data_accessed": False,
        "test_and_challenge_accessed": False,
        "dataset": str(frozen["asset"]),
        "model": args.model,
        "adapter": args.adapter,
        "case_count": len(records),
        "full_frozen_bank": args.limit is None,
        "protocol": protocol,
        **fingerprints,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "peft": package_version("peft"),
            "cuda": torch.version.cuda,
        },
    }
    evaluation_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cap = mem.apply_cap(args.cap_mib if torch.cuda.is_available() else None)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload().to(device).eval()
    model.config.use_cache = True

    pending = [record for record in records if str(record["id"]) not in done]
    print(f"dataset   : {dataset_path.name} ({len(records):,} cases)")
    print(f"adapter   : {args.adapter}")
    print(f"device    : {device} cap {cap if cap else 'n/a'} MiB")
    print(f"protocol  : batch={args.batch_size} max_new_tokens={frozen['generation']['max_new_tokens']} greedy")
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset:offset + args.batch_size]
        rendered = [tokenizer.apply_chat_template(
            [{"role": "user", "content": str(record["prompt"])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for record in batch]
        inputs = tokenizer(rendered, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=int(frozen["generation"]["max_new_tokens"]),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_width = int(inputs.input_ids.shape[1])
        new_tokens = generated[:, prompt_width:]
        rows = []
        for index, record in enumerate(batch):
            raw = tokenizer.decode(new_tokens[index], skip_special_tokens=False).strip()
            prediction = tokenizer.decode(new_tokens[index], skip_special_tokens=True).strip()
            scored = score_prediction(record, prediction)
            row = {
                "id": str(record["id"]),
                "family": str(record["family"]),
                "difficulty": str(record["difficulty"]),
                "answer_type": str(record.get("answer_type", "unknown")),
                "prediction": prediction,
                "raw_prediction": raw,
                "prompt_tokens": int(inputs.attention_mask[index].sum().item()),
                "generated_tensor_tokens": int(new_tokens[index].numel()),
                **scored,
            }
            rows.append(row)
            done[row["id"]] = row
        with results_path.open("a", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        completed = len(done)
        elapsed = time.time() - started
        with progress_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({
                "done": completed,
                "total": len(records),
                "elapsed_s_this_process": elapsed,
                **mem.snapshot(),
            }) + "\n")
        if completed % 256 < len(batch) or completed == len(records):
            print(f"  {completed:>4,d}/{len(records):,} ({100 * completed / len(records):5.1f}%) {elapsed:,.1f}s")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - started
    if len(done) != len(records):
        raise SystemExit("capability run ended without full coverage")
    summary = summarize(records, done)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run_manifest.update({
        "status": "COMPLETE",
        "elapsed_s_this_process": elapsed,
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / (1 << 20)
            if device.type == "cuda" else None),
        "peak_reserved_mib": (
            torch.cuda.max_memory_reserved(device) / (1 << 20)
            if device.type == "cuda" else None),
        "accuracy": summary["accuracy"],
        "correct": summary["correct"],
    })
    evaluation_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
