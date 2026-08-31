"""Local-only Qwen/Rineya graft training runner.

This runner intentionally stops at unsealed train/dev loss evidence.  It does
not call the repository's sealed or showcase evaluators and cannot promote a
candidate.  Matched semantic-gate runs remain a separate preregistered phase.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .config import ExperimentConfig
from .evidence import (
    build_manifest,
    fingerprint_model_dir,
    fingerprint_paths,
    sha256_file,
    atomic_write_json,
)
from .factory import build_graft_branch, component_plan
from .graft import freeze_module, parameter_counts
from .integration import attach_grafts, model_signature, save_graft_pack




def _verify_split_manifest(
    manifest_path: str | Path,
    train_path: str | Path,
    dev_path: str | Path,
) -> dict[str, Any]:
    """Prove that the configured files are the frozen grouped split."""

    source = Path(manifest_path)
    record = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or int(record.get("schema_version", -1)) != 1:
        raise ValueError("unsupported or malformed split manifest")
    if int(record.get("group_overlap", -1)) != 0:
        raise ValueError("split manifest does not prove zero group overlap")
    for role, path in (("train", Path(train_path)), ("dev", Path(dev_path))):
        arm = record.get(role)
        if not isinstance(arm, dict):
            raise ValueError(f"split manifest is missing {role!r}")
        if int(arm.get("rows", 0)) <= 0 or int(arm.get("groups", 0)) <= 0:
            raise ValueError(f"split manifest {role!r} arm is empty")
        actual = sha256_file(path)
        if arm.get("sha256") != actual:
            raise ValueError(
                f"{role} data hash differs from the frozen split manifest: "
                f"expected {arm.get('sha256')}, got {actual}"
            )
    return {
        "path": source.as_posix(),
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "group_overlap": 0,
        "train_groups": int(record["train"]["groups"]),
        "dev_groups": int(record["dev"]["groups"]),
    }


def _fingerprint_data_assets(data_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for role, key in (("train", "train_mix"), ("dev", "dev_mix")):
        item = fingerprint_paths([data_cfg[key]])[0]
        item["role"] = role
        assets.append(item)
    split = _verify_split_manifest(
        data_cfg["split_manifest"], data_cfg["train_mix"], data_cfg["dev_mix"]
    )
    split["role"] = "split_manifest"
    assets.append(split)
    return assets

def _dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name!r}") from exc


def _set_memory_cap(device: torch.device, cap_mib: int | None) -> None:
    if device.type != "cuda" or cap_mib is None:
        return
    total = torch.cuda.get_device_properties(device).total_memory
    fraction = (cap_mib * 1024**2) / total
    if not 0 < fraction < 1:
        raise ValueError(
            f"memory cap {cap_mib} MiB must be below device total {total / 1024**2:.0f} MiB"
        )
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)


def _load_training_dependencies():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "real mode requires transformers in the project environment"
        ) from exc
    try:
        from scripts.train import dataset as dataset_module
    except ImportError as exc:
        raise RuntimeError(
            "run real mode from the repository root so scripts.train.dataset is importable"
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer, dataset_module


def _collate_to_device(dataset_module, data, indices, tokenizer, max_len, device):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    arrays = dataset_module.collate(data, indices, pad_id, max_len)
    return tuple(torch.from_numpy(array).to(device=device) for array in arrays)


def _probe_logits(model: nn.Module, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
    model.eval()
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return output.logits.detach().cpu()


def _frozen_versions(model: nn.Module) -> dict[str, int]:
    return {
        name: parameter._version
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }


def _check_frozen_versions(model: nn.Module, before: dict[str, int]) -> tuple[bool, list[str]]:
    after = dict(model.named_parameters())
    changed = [
        name
        for name, version in before.items()
        if name not in after or after[name]._version != version
    ]
    return not changed, changed


def _evaluate_nll(
    model: nn.Module,
    data,
    dataset_module,
    tokenizer,
    *,
    max_len: int,
    max_rows: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    category_loss: dict[str, float] = defaultdict(float)
    category_tokens: dict[str, int] = defaultdict(int)
    rows = min(len(data), max_rows)
    with torch.no_grad():
        for index in range(rows):
            input_ids, attention, labels = _collate_to_device(
                dataset_module, data, [index], tokenizer, max_len, device
            )
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                labels=labels,
                use_cache=False,
            )
            supervised = int((labels[:, 1:] != dataset_module.IGNORE_INDEX).sum().item())
            if supervised == 0:
                continue
            loss = float(output.loss.detach().float().cpu())
            total_loss += loss * supervised
            total_tokens += supervised
            category = data.categories[index]
            category_loss[category] += loss * supervised
            category_tokens[category] += supervised
    if total_tokens == 0:
        raise RuntimeError("dev split contains no supervised tokens")
    mean_nll = total_loss / total_tokens
    return {
        "rows": rows,
        "supervised_tokens": total_tokens,
        "mean_nll": mean_nll,
        "perplexity": math.exp(min(mean_nll, 20.0)),
        "per_category_nll": {
            category: category_loss[category] / category_tokens[category]
            for category in sorted(category_tokens)
        },
    }


def run_real(config: ExperimentConfig) -> tuple[dict[str, Any], Path]:
    if config.mode != "real":
        raise ValueError("run_real requires mode='real'")
    raw = config.raw
    AutoModelForCausalLM, AutoTokenizer, dataset_module = _load_training_dependencies()

    seed = config.seed
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    host_cfg = raw["host"]
    device = torch.device(str(host_cfg.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _set_memory_cap(device, raw.get("memory", {}).get("cap_mib"))

    model_path = Path(str(host_cfg["model_path"]))
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"CAG v1 real mode is local-only; checkpoint directory not found: {model_path}"
        )
    dtype = _dtype(str(host_cfg.get("dtype", "bfloat16")))
    load_kwargs: dict[str, Any] = {
        "local_files_only": bool(host_cfg.get("local_files_only", True)),
        "torch_dtype": dtype,
    }
    if host_cfg.get("attn_implementation"):
        load_kwargs["attn_implementation"] = host_cfg["attn_implementation"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.to(device)

    actual_hidden = int(getattr(model.config, "hidden_size"))
    declared_hidden = int(host_cfg["hidden_size"])
    if actual_hidden != declared_hidden:
        raise ValueError(
            f"host.hidden_size={declared_hidden} but checkpoint reports {actual_hidden}"
        )

    data_cfg = raw["data"]
    data_assets = _fingerprint_data_assets(data_cfg)
    max_len = int(raw.get("training", {}).get("max_len", 512))
    cache_dir = Path(str(data_cfg["cache_dir"]))
    train_data = dataset_module.build(
        Path(str(data_cfg["train_mix"])),
        tokenizer,
        max_len=max_len,
        cache_dir=cache_dir,
        tokenizer_dir=model_path,
    )
    dev_data = dataset_module.build(
        Path(str(data_cfg["dev_mix"])),
        tokenizer,
        max_len=max_len,
        cache_dir=cache_dir,
        tokenizer_dir=model_path,
    )
    max_train_rows = min(
        len(train_data), int(data_cfg.get("max_train_rows", len(train_data)))
    )
    if max_train_rows <= 0:
        raise RuntimeError("train split is empty")
    max_dev_rows = min(
        len(dev_data), int(data_cfg.get("max_dev_rows", len(dev_data)))
    )
    if max_dev_rows <= 0:
        raise RuntimeError("dev split is empty")

    baseline_dev = _evaluate_nll(
        model,
        dev_data,
        dataset_module,
        tokenizer,
        max_len=max_len,
        max_rows=max_dev_rows,
        device=device,
    )
    requested_probe_rows = int(
        raw.get("evaluation", {}).get("equivalence_probe_rows", 8)
    )
    probe_count = min(requested_probe_rows, max_train_rows)
    probe_indices = [
        min(max_train_rows - 1, (index * max_train_rows) // probe_count)
        for index in range(probe_count)
    ]
    probe_ids, probe_attention, _ = _collate_to_device(
        dataset_module, train_data, probe_indices, tokenizer, max_len, device
    )
    logits_before = _probe_logits(model, probe_ids, probe_attention)
    host_parameter_count = sum(parameter.numel() for parameter in model.parameters())

    freeze_module(model)
    attachments = attach_grafts(
        model,
        host_cfg["layer_indices"],
        lambda _index, _layer: build_graft_branch(config, host_dim=actual_hidden),
        layer_path=host_cfg.get("layer_path"),
        freeze_host=bool(raw["graft"].get("freeze_host", True)),
        freeze_donor=bool(raw["graft"].get("freeze_donor", False)),
        exact_eval_bypass=bool(raw["graft"].get("exact_eval_bypass", True)),
    )
    logits_after = _probe_logits(model, probe_ids, probe_attention)
    zero_gate_bit_exact = torch.equal(logits_before, logits_after)
    zero_gate_max_abs = float((logits_before - logits_after).abs().max())
    if not zero_gate_bit_exact:
        raise RuntimeError(
            f"zero-gate equivalence failed: max |delta logits|={zero_gate_max_abs:.3e}"
        )

    trainable_named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    forbidden_trainable = [name for name, _ in trainable_named if ".graft_branch." not in name]
    if forbidden_trainable:
        raise RuntimeError(f"non-graft parameters are trainable: {forbidden_trainable[:8]}")
    if not trainable_named:
        raise RuntimeError("no graft parameters are trainable")
    frozen_versions = _frozen_versions(model)

    train_cfg = raw.get("training", {})
    steps = int(train_cfg.get("steps", 100))
    batch_size = int(train_cfg.get("batch_size", 1))
    grad_accum = int(train_cfg.get("grad_accum", 1))
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    optimizer = torch.optim.AdamW(
        [p for _, p in trainable_named], lr=lr, weight_decay=weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    order = torch.randperm(max_train_rows, generator=generator).tolist()
    cursor = 0
    losses: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    train_started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(steps):
        accumulated = 0.0
        for _ in range(grad_accum):
            if cursor + batch_size > len(order):
                order = torch.randperm(max_train_rows, generator=generator).tolist()
                cursor = 0
            indices = order[cursor : cursor + batch_size]
            cursor += batch_size
            input_ids, attention, labels = _collate_to_device(
                dataset_module, train_data, indices, tokenizer, max_len, device
            )
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                labels=labels,
                use_cache=False,
            )
            loss = output.loss / grad_accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward()
            accumulated += float(loss.detach().float().cpu())
        nn.utils.clip_grad_norm_([p for _, p in trainable_named], grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(accumulated)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - train_started
    peak_allocated_mib = (
        torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
    )
    peak_reserved_mib = (
        torch.cuda.max_memory_reserved(device) / 1024**2 if device.type == "cuda" else None
    )

    frozen_unchanged, changed_frozen = _check_frozen_versions(model, frozen_versions)
    if not frozen_unchanged:
        raise RuntimeError(f"frozen parameters changed: {changed_frozen[:8]}")

    dev_metrics = _evaluate_nll(
        model,
        dev_data,
        dataset_module,
        tokenizer,
        max_len=max_len,
        max_rows=max_dev_rows,
        device=device,
    )

    run_dir = Path(str(raw.get("output", {}).get("run_dir", "artifacts/cag_run")))
    if run_dir.exists() and any(run_dir.iterdir()) and not raw.get("output", {}).get(
        "overwrite", False
    ):
        raise FileExistsError(f"run directory is non-empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    model_asset = fingerprint_model_dir(model_path)
    graft_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".graft_branch." in name
    )
    graft_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".graft_branch." in name and parameter.requires_grad
    )
    arm = {
        "name": config.experiment_id,
        **parameter_counts(model),
        "host_parameters": host_parameter_count,
        "graft_parameters": graft_parameters,
        "graft_trainable_parameters": graft_trainable,
        "component_plan": component_plan(config),
        "attachments": [attachment.__dict__ for attachment in attachments],
        "model_signature": model_signature(model, host_cfg.get("layer_path")),
    }
    results = {
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "steps": steps,
        "train_seconds": train_seconds,
        "peak_allocated_mib": peak_allocated_mib,
        "peak_reserved_mib": peak_reserved_mib,
        "equivalence_probe_rows": probe_count,
        "equivalence_probe_indices": probe_indices,
        "dev_before": baseline_dev,
        "dev_after": dev_metrics,
        "screening_interpretation": (
            "Train/dev NLL is a health screen only. Semantic capability promotion "
            "requires the separately frozen open-development gates in the protocol."
        ),
    }
    invariants = {
        "zero_gate_bit_exact": zero_gate_bit_exact,
        "zero_gate_max_abs_error": zero_gate_max_abs,
        "only_graft_parameters_trainable": not forbidden_trainable,
        "frozen_parameters_unchanged": frozen_unchanged,
        "sealed_evaluation_opened": False,
    }
    manifest = build_manifest(
        config=config,
        status="UNSEALED_MECHANISM_RUN_COMPLETE",
        phase="real_train_dev",
        results=results,
        arms=[arm],
        invariants=invariants,
        data_assets=data_assets,
        model_assets=[model_asset],
    )
    manifest_path = atomic_write_json(run_dir / "evidence_manifest.json", manifest)
    save_graft_pack(
        model,
        run_dir / "graft_pack",
        {
            "experiment_id": config.experiment_id,
            "config_sha256": config.sha256,
            "host_model_aggregate_sha256": model_asset["aggregate_sha256"],
            "attachments": [attachment.__dict__ for attachment in attachments],
            "evidence_manifest": manifest_path.name,
        },
    )
    return manifest, manifest_path
