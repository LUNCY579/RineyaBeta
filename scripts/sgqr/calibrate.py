#!/usr/bin/env python3
"""Calibrate SGQR group significance from paired dense/deleted/group forwards.

This workflow intentionally contains no confidence lens or confidence threshold.
It measures the fundamental question: which frozen quantized donor groups reduce
the discrepancy caused by deleting the target MLP?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

import memory as mem  # noqa: E402
from scripts.sgqr.install import install_sgqr_mlp  # noqa: E402
from scripts.sgqr.utility import (  # noqa: E402
    SignificanceAccumulator,
    marginal_recovery_utility,
    teacher_kl,
)

DATASETS = {
    "chat_dev": "artifacts/eval_sets/chat_dev.jsonl",
    "rag_dev": "artifacts/eval_sets/rag_dev.jsonl",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_messages(record: dict) -> list[dict]:
    messages = [dict(message) for message in record.get("messages") or []]
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    if not messages:
        raise ValueError(f"record {record.get('id', '?')} has no prompt messages")
    return messages


def last_logits(model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    kwargs = dict(inputs)
    kwargs["use_cache"] = False
    try:
        outputs = model(**kwargs, logits_to_keep=1)
    except TypeError:
        outputs = model(**kwargs)
    return outputs.logits[:, -1, :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/rineya-core-en32k-28L")
    parser.add_argument("--dataset", default="chat_dev")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--cap-mib", type=float, default=5120.0)
    parser.add_argument("--exposure-exponent", type=float, default=1.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_path = ROOT / args.model
    dataset_path = ROOT / DATASETS.get(args.dataset, args.dataset)
    out_dir = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = records[: args.limit]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.bfloat16 if use_cuda else torch.float32
    cap = mem.apply_cap(args.cap_mib if use_cuda else None)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    model.config.use_cache = False

    # Build one paired intervention. The dense donor is retained only for this
    # calibration process; deployment artifacts drop it.
    installation = install_sgqr_mlp(
        model,
        layer_idx=args.layer,
        mode="quantized_all",
        bits=args.bits,
        group_size=args.group_size,
        retain_dense_reference=True,
        random_seed=args.seed,
    )
    if installation.wrapper is None:
        raise RuntimeError("calibration expected an SGQR wrapper")
    wrapper = installation.wrapper
    wrapper.capture_last_input = True
    model = model.to(device).eval()

    accumulator = SignificanceAccumulator(wrapper.remnant.group_count)
    features: list[torch.Tensor] = []
    utilities: list[torch.Tensor] = []
    deleted_kls: list[float] = []
    full_utilities: list[float] = []
    trace_rows: list[dict] = []

    started = time.time()
    for index, record in enumerate(records):
        messages = normalize_messages(record)
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(rendered, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            wrapper.set_mode("dense")
            teacher = last_logits(model, inputs)
            if wrapper.last_input is None:
                raise RuntimeError("failed to capture pre-remnant MLP input")
            feature = wrapper.last_input[:, -1, :].detach().float().cpu()

            wrapper.set_mode("deleted")
            deleted = last_logits(model, inputs)
            base_kl = teacher_kl(teacher, deleted, reduction="mean")

            wrapper.set_mode("quantized_all")
            full = last_logits(model, inputs)
            full_utility = marginal_recovery_utility(
                teacher, deleted, full
            ).mean()

            group_values = []
            wrapper.set_mode("fixed_groups")
            for group_id in range(wrapper.remnant.group_count):
                wrapper.set_fixed_groups([group_id])
                candidate = last_logits(model, inputs)
                utility = marginal_recovery_utility(
                    teacher, deleted, candidate
                ).mean()
                group_values.append(float(utility.item()))

        utility_tensor = torch.tensor(group_values, dtype=torch.float32)
        accumulator.update(utility_tensor)
        features.append(feature.squeeze(0).to(torch.float16))
        utilities.append(utility_tensor.to(torch.float32))
        deleted_kls.append(float(base_kl.item()))
        full_utilities.append(float(full_utility.item()))
        trace_rows.append(
            {
                "id": str(record.get("id", index)),
                "prompt_tokens": int(inputs["input_ids"].shape[1]),
                "deleted_teacher_kl": float(base_kl.item()),
                "full_remnant_utility": float(full_utility.item()),
                "group_utilities": group_values,
            }
        )

        if (index + 1) % 5 == 0 or index + 1 == len(records):
            print(
                f"{index + 1:>4}/{len(records)} prompts, "
                f"{time.time() - started:.1f}s elapsed"
            )

    significance = accumulator.summary()
    trace_path = out_dir / "utility_trace.pt"
    torch.save(
        {
            "features": torch.stack(features),
            "utilities": torch.stack(utilities),
            "metadata": {
                "format": "sgqr-utility-trace-v1",
                "layer_idx": args.layer,
                "bits": args.bits,
                "group_size": args.group_size,
                "group_count": wrapper.remnant.group_count,
                "hidden_size": wrapper.remnant.hidden_size,
                "dataset": str(dataset_path.relative_to(ROOT)).replace("\\", "/"),
                "seed": args.seed,
            },
        },
        trace_path,
    )
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in trace_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "format": "sgqr-calibration-summary-v1",
        "model": args.model,
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "dataset": str(dataset_path.relative_to(ROOT)).replace("\\", "/"),
        "dataset_sha256": sha256_file(dataset_path),
        "record_count": len(records),
        "layer_idx": args.layer,
        "bits": args.bits,
        "group_size": args.group_size,
        "group_count": wrapper.remnant.group_count,
        "exposure_exponent": args.exposure_exponent,
        "mean_deleted_teacher_kl": float(np.mean(deleted_kls)),
        "mean_full_remnant_utility": float(np.mean(full_utilities)),
        "positive_full_remnant_fraction": float(np.mean(np.asarray(full_utilities) > 0)),
        "significance": significance.as_dict(args.exposure_exponent),
        "magnitude_scores": wrapper.remnant.magnitude_scores.detach().cpu().tolist(),
        "group_costs": wrapper.remnant.group_costs.detach().cpu().tolist(),
        "storage": installation.storage_manifest(),
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if use_cuda else None,
            "cap_mib": cap,
            "torch": torch.__version__,
        },
        "elapsed_s": round(time.time() - started, 3),
        "trace_file": trace_path.name,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"\nwrote {out_dir}")
    print(f"mean deleted KL        : {summary['mean_deleted_teacher_kl']:.6f}")
    print(f"mean full utility      : {summary['mean_full_remnant_utility']:+.6f}")
    print(f"positive full fraction : {summary['positive_full_remnant_fraction']:.3f}")


if __name__ == "__main__":
    main()
