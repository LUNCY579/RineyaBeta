#!/usr/bin/env python3
"""Executable leakage-resistant one-step real-gradient evidence run for PCSM Phase C."""

from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "train"))

from peer_features import LayerMetadata, LayerView  # noqa: E402
from peer_stabilizer import PeerStabilizerMatrix  # noqa: E402
from rineya_integration import resolve_submodule, transformer_layers  # noqa: E402
from phase_c_controls import (  # noqa: E402
    CompactPlan,
    apply_first_adamw_step,
    coordinate_shuffled_views,
    gradient_topk_support,
    layer_shuffled_views,
    random_support,
    stable_u64,
    support_from_scores,
    support_jaccard,
    target_only_scores,
    tensor_sha256,
)
from phase_c_report import hash_evidence_tree, summarize, write_decision_markdown  # noqa: E402
from phase_c_splits import (  # noqa: E402
    IndexedRow,
    index_sources,
    select_rows,
    selection_manifest,
    sha256_file,
)

IGNORE_INDEX = -100


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Phase-C config must be a mapping")
    cfg["_config_path"] = str(config_path.resolve())
    return cfg


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def seed_everything(seed: int, *, strict: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if strict:
        torch.use_deterministic_algorithms(True, warn_only=False)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=REPO, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain") or ""),
    }


def environment_manifest() -> dict[str, Any]:
    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = None
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def resolve_device(cfg: Mapping[str, Any]) -> torch.device:
    requested = str(cfg["model"].get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype {name!r}")
    dtype = mapping[name]
    if device.type == "cpu" and dtype != torch.float32:
        return torch.float32
    return dtype


def checkpoint_ledger(model_dir: Path, *, hash_weights: bool) -> dict[str, Any]:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_dir}")
    metadata_names = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    )
    files: list[dict[str, Any]] = []
    for name in metadata_names:
        path = model_dir / name
        if path.exists():
            files.append({"path": name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    weight_files = sorted(model_dir.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(
            f"no local .safetensors checkpoint shards found in {model_dir}; "
            "Phase C forbids implicit remote checkpoint substitution"
        )
    for path in weight_files:
        files.append(
            {
                "path": path.name,
                "sha256": sha256_file(path) if hash_weights else "not_hashed",
                "bytes": path.stat().st_size,
            }
        )
    ledger = {"model_dir": str(model_dir), "files": files}
    ledger["checkpoint_sha256"] = sha256_json(ledger)
    return ledger


def load_model_and_tokenizer(cfg: Mapping[str, Any], device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = REPO / cfg["model"]["path"]
    dtype = resolve_dtype(str(cfg["model"].get("dtype", "bfloat16")), device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=bool(cfg["model"].get("local_files_only", True)),
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=dtype,
        attn_implementation=str(cfg["model"].get("attn_implementation", "eager")),
        local_files_only=bool(cfg["model"].get("local_files_only", True)),
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
    )
    model.config.use_cache = False
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model


def candidate_parameters(model: nn.Module, module_path: str) -> list[tuple[int, str, nn.Parameter]]:
    name_map = {id(parameter): name for name, parameter in model.named_parameters()}
    rows: list[tuple[int, str, nn.Parameter]] = []
    for depth, layer in enumerate(transformer_layers(model)):
        module = resolve_submodule(layer, module_path)
        parameter = module.weight
        if parameter.ndim != 2:
            raise ValueError(f"{module_path} layer {depth}: expected 2-D weight")
        rows.append((depth, name_map[id(parameter)], parameter))
    if len(rows) < 2:
        raise RuntimeError("PCSM requires at least two retained layers in the module family")
    return rows


def candidate_hash(candidates: Sequence[tuple[int, str, nn.Parameter]]) -> str:
    payload = [
        {
            "depth": depth,
            "name": name,
            "shape": list(parameter.shape),
            "sha256": tensor_sha256(parameter),
        }
        for depth, name, parameter in candidates
    ]
    return sha256_json(payload)


def snapshot_candidates(
    candidates: Sequence[tuple[int, str, nn.Parameter]]
) -> dict[str, Tensor]:
    return {name: parameter.detach().cpu().clone() for _, name, parameter in candidates}


@torch.no_grad()
def restore_candidates(
    candidates: Sequence[tuple[int, str, nn.Parameter]], snapshot: Mapping[str, Tensor]
) -> None:
    for _, name, parameter in candidates:
        parameter.copy_(snapshot[name].to(parameter.device, dtype=parameter.dtype))


def _render_row(tokenizer, row: IndexedRow, max_len: int) -> dict[str, Tensor | int]:
    messages = list(row.messages)
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    eos = tokenizer.eos_token or "<|im_end|>"
    target_text = str(messages[-1]["content"]) + eos
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    target_ids = tokenizer(target_text, add_special_tokens=False).input_ids
    if len(prompt_ids) + len(target_ids) > max_len:
        room = max_len - len(target_ids)
        if room <= 0:
            target_ids = target_ids[:max_len]
            prompt_ids = []
        else:
            prompt_ids = prompt_ids[-room:]
    ids = prompt_ids + target_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids
    if len(ids) < 2 or all(value == IGNORE_INDEX for value in labels[1:]):
        raise ValueError(f"row {row.row_id} has no supervised causal target after truncation")
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention = torch.ones_like(input_ids)
    label_tensor = torch.tensor([labels], dtype=torch.long)
    supervised = int((label_tensor[:, 1:] != IGNORE_INDEX).sum().item())
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "labels": label_tensor,
        "supervised_tokens": supervised,
    }


def tokenize_rows(tokenizer, rows: Sequence[IndexedRow], max_len: int) -> list[dict[str, Any]]:
    return [_render_row(tokenizer, row, max_len) for row in rows]


def _loss_sum(model: nn.Module, batch: Mapping[str, Any], device: torch.device) -> tuple[Tensor, int]:
    input_ids = batch["input_ids"].to(device)
    attention = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    logits = model(input_ids=input_ids, attention_mask=attention).logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    loss_sum = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    counted = int((shift_labels != IGNORE_INDEX).sum().item())
    return loss_sum, counted


@torch.no_grad()
def evaluate_loss(model: nn.Module, batches: Sequence[Mapping[str, Any]], device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in batches:
        loss_sum, counted = _loss_sum(model, batch, device)
        total_loss += float(loss_sum.item())
        total_tokens += counted
    if total_tokens == 0:
        raise RuntimeError("evaluation has zero supervised tokens")
    return total_loss / total_tokens


def calibration_gradients(
    model: nn.Module,
    candidates: Sequence[tuple[int, str, nn.Parameter]],
    batches: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> dict[str, Tensor]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for _, _, parameter in candidates:
        parameter.requires_grad_(True)
    total_tokens = sum(int(batch["supervised_tokens"]) for batch in batches)
    if total_tokens <= 0:
        raise RuntimeError("MASK_CAL has zero supervised tokens")
    model.zero_grad(set_to_none=True)
    for batch in batches:
        loss_sum, _ = _loss_sum(model, batch, device)
        (loss_sum / total_tokens).backward()
    gradients: dict[str, Tensor] = {}
    for _, name, parameter in candidates:
        if parameter.grad is None:
            raise RuntimeError(f"MASK_CAL produced no gradient for {name}")
        gradients[name] = parameter.grad.detach().float().cpu().clone()
        parameter.grad = None
        parameter.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    return gradients


def make_views(
    candidates: Sequence[tuple[int, str, nn.Parameter]],
    *,
    target_index: int,
    target_gradient: Tensor,
    module_path: str,
) -> list[LayerView]:
    views: list[LayerView] = []
    for index, (depth, name, parameter) in enumerate(candidates):
        gradient = (
            target_gradient.to(parameter.device, dtype=parameter.dtype)
            if index == target_index
            else None
        )
        views.append(
            LayerView(
                weight=parameter.detach(),
                gradient=gradient,
                metadata=LayerMetadata(name=name, depth=depth, module_type=module_path),
            )
        )
    return views


def generator_state_sha256(generator: nn.Module) -> str:
    payload = [
        (name, tensor_sha256(tensor))
        for name, tensor in sorted(generator.state_dict().items())
    ]
    return sha256_json(payload)


def _functional_first_adam_weight(
    weight: Tensor,
    gradient: Tensor,
    training_mask: Tensor,
    step_scale: Tensor,
    *,
    lr: float,
    eps: float,
) -> Tensor:
    grad = gradient.float()
    update = grad / (grad.abs() + eps)
    updated = weight.float() - lr * training_mask.float() * step_scale.float() * update
    return updated.to(weight.dtype)


def meta_calibrate_generator(
    model: nn.Module,
    generator: PeerStabilizerMatrix,
    candidates: Sequence[tuple[int, str, nn.Parameter]],
    batches: Sequence[Mapping[str, Any]],
    *,
    cfg: Mapping[str, Any],
    seed: int,
    module_path: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    """First-order meta-calibration using META_TRAIN only."""
    from torch.func import functional_call

    meta = cfg["pcsm"]["meta"]
    steps = int(meta["steps"])
    if steps <= 0:
        return []
    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=float(meta["optimizer_lr"]),
        weight_decay=float(meta.get("weight_decay", 0.0)),
    )
    curve: list[dict[str, Any]] = []
    generator.train()
    for step in range(steps):
        target_index = int(
            stable_u64(seed, int(meta.get("schedule_seed_offset", 70001)), step)
            % len(candidates)
        )
        inner_batch = batches[(2 * step) % len(batches)]
        outer_batch = batches[(2 * step + 1) % len(batches)]
        depth, parameter_name, parameter = candidates[target_index]

        for p in model.parameters():
            p.requires_grad_(False)
        parameter.requires_grad_(True)
        inner_sum, inner_tokens = _loss_sum(model, inner_batch, device)
        inner_loss = inner_sum / max(inner_tokens, 1)
        inner_gradient = torch.autograd.grad(inner_loss, parameter, create_graph=False)[0].detach()
        parameter.requires_grad_(False)

        views = make_views(
            candidates,
            target_index=target_index,
            target_gradient=inner_gradient,
            module_path=module_path,
        )
        output = generator(
            views,
            target_index,
            cfg["pcsm"]["budget"],
            task_gradient=inner_gradient,
            straight_through=True,
            mask_temperature=float(meta.get("mask_temperature", 0.30)),
        )
        updated_weight = _functional_first_adam_weight(
            parameter.detach(),
            inner_gradient,
            output.training_mask,
            output.step_scale,
            lr=float(meta["inner_step_lr"]),
            eps=float(cfg["optimizer"]["eps"]),
        )

        optimizer.zero_grad(set_to_none=True)
        inputs = {
            "input_ids": outer_batch["input_ids"].to(device),
            "attention_mask": outer_batch["attention_mask"].to(device),
        }
        labels = outer_batch["labels"].to(device)
        logits = functional_call(
            model,
            {parameter_name: updated_weight},
            (),
            inputs,
            strict=False,
        ).logits
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        outer_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="mean",
        )
        if not bool(torch.isfinite(outer_loss)):
            raise RuntimeError(f"non-finite meta loss at step {step}")
        outer_loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                generator.parameters(), float(meta.get("grad_clip", 5.0))
            ).item()
        )
        if not any(p.grad is not None for p in generator.parameters()):
            raise RuntimeError("meta-calibration produced no generator gradients")
        optimizer.step()
        curve.append(
            {
                "step": step,
                "target_depth": depth,
                "inner_loss": float(inner_loss.detach().item()),
                "outer_loss": float(outer_loss.detach().item()),
                "generator_grad_norm": grad_norm,
            }
        )
    generator.eval()
    return curve


def build_pcsm_plan(
    model: nn.Module,
    generator: PeerStabilizerMatrix,
    candidates: Sequence[tuple[int, str, nn.Parameter]],
    gradients: Mapping[str, Tensor],
    *,
    arm: str,
    budget: int | float,
    module_path: str,
    seed: int,
    transform: str,
) -> CompactPlan:
    supports = []
    transform_rows: list[dict[str, Any]] = []
    generator.eval()
    for target_index, (depth, name, parameter) in enumerate(candidates):
        target_gradient = gradients[name].to(parameter.device, dtype=parameter.dtype)
        views = make_views(
            candidates,
            target_index=target_index,
            target_gradient=target_gradient,
            module_path=module_path,
        )
        if transform == "coordinate_shuffle":
            views, ledger = coordinate_shuffled_views(
                views, target_index=target_index, seed=seed + depth * 1009
            )
        elif transform == "layer_shuffle":
            views, ledger = layer_shuffled_views(
                views, target_index=target_index, seed=seed + depth * 1009
            )
        elif transform == "real":
            ledger = {"kind": "real_peers"}
        else:
            raise ValueError(f"unknown transform {transform!r}")

        with torch.no_grad():
            output = generator(
                views,
                target_index,
                budget,
                task_gradient=target_gradient,
                straight_through=False,
            )
        supports.append(
            support_from_scores(
                parameter_name=name,
                depth=depth,
                scores=output.scores,
                budget=budget,
                scales=output.step_scale,
                attention=output.attention,
                feature_weights=output.feature_weights,
            )
        )
        transform_rows.append({"target_depth": depth, **ledger})
        del output, views
        if parameter.device.type == "cuda":
            torch.cuda.empty_cache()
    return CompactPlan(
        arm=arm,
        budget=budget,
        module_path=module_path,
        supports=supports,
        transform={"kind": transform, "per_target": transform_rows},
        generator_sha256=generator_state_sha256(generator),
    )


def build_baseline_plan(
    candidates: Sequence[tuple[int, str, nn.Parameter]],
    gradients: Mapping[str, Tensor],
    *,
    arm: str,
    budget: int | float,
    module_path: str,
    seed: int,
    common_scale: float,
) -> CompactPlan:
    supports = []
    for depth, name, parameter in candidates:
        gradient = gradients[name]
        if arm == "random":
            support = random_support(
                parameter,
                parameter_name=name,
                depth=depth,
                budget=budget,
                seed=seed,
                scale=common_scale,
            )
        elif arm == "gradient_topk":
            support = gradient_topk_support(
                gradient,
                parameter_name=name,
                depth=depth,
                budget=budget,
                scale=common_scale,
            )
        elif arm == "target_only":
            scores = target_only_scores(parameter.detach().cpu(), gradient)
            support = support_from_scores(
                parameter_name=name,
                depth=depth,
                scores=scores,
                budget=budget,
                scales=common_scale,
            )
        else:
            raise ValueError(arm)
        supports.append(support)
    return CompactPlan(
        arm=arm,
        budget=budget,
        module_path=module_path,
        supports=supports,
        transform={"kind": arm},
        generator_sha256=None,
    )


def save_plan(plan: CompactPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": plan.manifest(),
        "supports": [
            {
                "parameter_name": support.parameter_name,
                "shape": support.shape,
                "depth": support.depth,
                "indices": support.indices,
                "scales": support.scales,
                "mask_sha256": support.mask_sha256,
                "attention": support.attention,
                "feature_weights": support.feature_weights,
            }
            for support in plan.supports
        ],
    }
    torch.save(payload, path)


def prepare(cfg: Mapping[str, Any]) -> tuple[list[IndexedRow], dict[str, Any], Path]:
    source_paths = list(cfg["data"]["source_files"])
    expected = cfg["data"].get("expected_source_sha256", {})
    indexed, split_audit = index_sources(
        source_paths,
        root=REPO,
        group_fields=list(cfg["data"]["group_fields"]),
        split_seed=int(cfg["data"]["split"]["seed"]),
        ratios=cfg["data"]["split"]["ratios"],
        expected_sha256=expected,
    )
    out_dir = REPO / cfg["output"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    selection_audit: dict[str, Any] = {}
    for seed in cfg["experiment"]["seeds"]:
        selection_audit[str(seed)] = {}
        for split, count_key in (
            ("META_TRAIN", "meta_train_rows"),
            ("MASK_CAL", "mask_cal_rows"),
            ("ONE_STEP_DEV", "one_step_dev_rows"),
        ):
            rows = select_rows(
                indexed,
                split=split,
                count=int(cfg["data"][count_key]),
                seed=int(seed),
            )
            selection_audit[str(seed)][split] = selection_manifest(rows)

    payload = {
        "split_audit": split_audit,
        "selections": selection_audit,
        "sealed_used_for_model_computation": False,
    }
    payload["manifest_sha256"] = sha256_json(payload)
    (out_dir / "SPLIT_AUDIT.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return indexed, payload, out_dir


def trial_validity(
    *,
    pre_loss: float,
    post_loss: float,
    plan: CompactPlan,
    reference_count_signature: str,
    base_hash_before: str,
    expected_base_hash: str,
    audit: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not math.isfinite(pre_loss) or not math.isfinite(post_loss):
        reasons.append("non_finite_loss")
    if plan.count_signature != reference_count_signature:
        reasons.append("support_count_mismatch")
    if base_hash_before != expected_base_hash:
        reasons.append("base_candidate_hash_mismatch")
    if int(audit.get("changed_outside", -1)) != 0:
        reasons.append("outside_support_movement")
    return not reasons, reasons


def run_seed(
    *,
    cfg: Mapping[str, Any],
    seed: int,
    indexed: Sequence[IndexedRow],
    tokenizer,
    model: nn.Module,
    candidates: Sequence[tuple[int, str, nn.Parameter]],
    base_snapshot: Mapping[str, Tensor],
    base_candidate_sha: str,
    device: torch.device,
    out_dir: Path,
) -> list[dict[str, Any]]:
    strict = bool(cfg["model"].get("strict_determinism", True))
    seed_everything(seed, strict=strict)
    data_cfg = cfg["data"]
    max_len = int(data_cfg["max_len"])

    meta_rows = select_rows(indexed, split="META_TRAIN", count=int(data_cfg["meta_train_rows"]), seed=seed)
    mask_rows = select_rows(indexed, split="MASK_CAL", count=int(data_cfg["mask_cal_rows"]), seed=seed)
    dev_rows = select_rows(indexed, split="ONE_STEP_DEV", count=int(data_cfg["one_step_dev_rows"]), seed=seed)
    meta_batches = tokenize_rows(tokenizer, meta_rows, max_len)
    mask_batches = tokenize_rows(tokenizer, mask_rows, max_len)
    dev_batches = tokenize_rows(tokenizer, dev_rows, max_len)

    restore_candidates(candidates, base_snapshot)
    if candidate_hash(candidates) != base_candidate_sha:
        raise RuntimeError("candidate checkpoint restoration failed before seed")

    generator_seed = seed + 104729
    seed_everything(generator_seed, strict=strict)
    pcsm_cfg = cfg["pcsm"]
    bootstrap = PeerStabilizerMatrix(
        hidden_dim=int(pcsm_cfg["hidden_dim"]),
        cosine_temperature=float(pcsm_cfg.get("cosine_temperature", 0.75)),
        min_step_scale=float(pcsm_cfg.get("min_step_scale", 0.20)),
        max_step_scale=float(pcsm_cfg.get("max_step_scale", 1.20)),
    ).to(device)
    bootstrap.eval()
    bootstrap_state = deepcopy(bootstrap.state_dict())
    bootstrap_sha = generator_state_sha256(bootstrap)

    calibrated = PeerStabilizerMatrix(
        hidden_dim=int(pcsm_cfg["hidden_dim"]),
        cosine_temperature=float(pcsm_cfg.get("cosine_temperature", 0.75)),
        min_step_scale=float(pcsm_cfg.get("min_step_scale", 0.20)),
        max_step_scale=float(pcsm_cfg.get("max_step_scale", 1.20)),
    ).to(device)
    calibrated.load_state_dict(bootstrap_state)
    meta_curve = meta_calibrate_generator(
        model,
        calibrated,
        candidates,
        meta_batches,
        cfg=cfg,
        seed=seed,
        module_path=str(pcsm_cfg["module_path"]),
        device=device,
    )
    calibrated_sha = generator_state_sha256(calibrated)
    generator_dir = out_dir / f"seed_{seed}" / "generators"
    generator_dir.mkdir(parents=True, exist_ok=True)
    torch.save(bootstrap_state, generator_dir / "bootstrap.pt")
    torch.save(calibrated.state_dict(), generator_dir / "calibrated.pt")
    (generator_dir / "manifest.json").write_text(
        json.dumps(
            {
                "bootstrap_sha256": bootstrap_sha,
                "calibrated_sha256": calibrated_sha,
                "meta_curve": meta_curve,
                "meta_train_selection": selection_manifest(meta_rows),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    restore_candidates(candidates, base_snapshot)
    gradients = calibration_gradients(model, candidates, mask_batches, device)
    gradient_manifest = {
        name: {"sha256": tensor_sha256(grad), "l2": float(torch.linalg.vector_norm(grad).item())}
        for name, grad in gradients.items()
    }
    seed_dir = out_dir / f"seed_{seed}"
    (seed_dir / "mask_cal_gradients.json").write_text(
        json.dumps(
            {
                "mask_cal_selection": selection_manifest(mask_rows),
                "gradients": gradient_manifest,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    budget = pcsm_cfg["budget"]
    module_path = str(pcsm_cfg["module_path"])
    common_scale = float(pcsm_cfg.get("common_step_scale", 1.0))
    plans: dict[str, CompactPlan] = {
        "random": build_baseline_plan(
            candidates, gradients, arm="random", budget=budget, module_path=module_path,
            seed=seed, common_scale=common_scale,
        ),
        "gradient_topk": build_baseline_plan(
            candidates, gradients, arm="gradient_topk", budget=budget, module_path=module_path,
            seed=seed, common_scale=common_scale,
        ),
        "target_only": build_baseline_plan(
            candidates, gradients, arm="target_only", budget=budget, module_path=module_path,
            seed=seed, common_scale=common_scale,
        ),
        "real_peers_bootstrap": build_pcsm_plan(
            model, bootstrap, candidates, gradients, arm="real_peers_bootstrap",
            budget=budget, module_path=module_path, seed=seed, transform="real",
        ),
        "shuffled_peers_calibrated": build_pcsm_plan(
            model, calibrated, candidates, gradients, arm="shuffled_peers_calibrated",
            budget=budget, module_path=module_path, seed=seed, transform="coordinate_shuffle",
        ),
        "real_peers_calibrated": build_pcsm_plan(
            model, calibrated, candidates, gradients, arm="real_peers_calibrated",
            budget=budget, module_path=module_path, seed=seed, transform="real",
        ),
        "layer_shuffled_peers_calibrated": build_pcsm_plan(
            model, calibrated, candidates, gradients, arm="layer_shuffled_peers_calibrated",
            budget=budget, module_path=module_path, seed=seed, transform="layer_shuffle",
        ),
    }

    required = list(cfg["experiment"]["arms"])
    if set(required) != set(plans):
        raise RuntimeError(f"config arms {required} do not match implemented arms {sorted(plans)}")
    count_signatures = {plan.count_signature for plan in plans.values()}
    if len(count_signatures) != 1:
        raise RuntimeError("matched sparse arms resolved different support counts")
    reference_count_signature = next(iter(count_signatures))

    plan_dir = seed_dir / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    for arm, plan in plans.items():
        save_plan(plan, plan_dir / f"{arm}.pt")
        (plan_dir / f"{arm}.manifest.json").write_text(
            json.dumps(plan.manifest(), indent=2, sort_keys=True), encoding="utf-8"
        )

    overlaps = {
        "real_vs_gradient_topk": support_jaccard(plans["real_peers_calibrated"], plans["gradient_topk"]),
        "real_vs_shuffled": support_jaccard(plans["real_peers_calibrated"], plans["shuffled_peers_calibrated"]),
        "real_vs_target_only": support_jaccard(plans["real_peers_calibrated"], plans["target_only"]),
    }
    (seed_dir / "support_overlaps.json").write_text(
        json.dumps(overlaps, indent=2, sort_keys=True), encoding="utf-8"
    )

    optimizer_cfg = cfg["optimizer"]
    trials: list[dict[str, Any]] = []
    causal = ["real_peers_calibrated", "shuffled_peers_calibrated"]
    if seed % 2:
        causal.reverse()
    arm_order = causal + [
        "layer_shuffled_peers_calibrated",
        "real_peers_bootstrap",
        "target_only",
        "gradient_topk",
        "random",
    ]
    for arm in arm_order:
        restore_candidates(candidates, base_snapshot)
        base_hash_before = candidate_hash(candidates)
        pre_loss = evaluate_loss(model, dev_batches, device)
        audit = apply_first_adamw_step(
            model,
            plans[arm],
            gradients,
            lr=float(optimizer_cfg["lr"]),
            betas=tuple(float(v) for v in optimizer_cfg["betas"]),
            eps=float(optimizer_cfg["eps"]),
            weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
        )
        post_loss = evaluate_loss(model, dev_batches, device)
        valid, reasons = trial_validity(
            pre_loss=pre_loss,
            post_loss=post_loss,
            plan=plans[arm],
            reference_count_signature=reference_count_signature,
            base_hash_before=base_hash_before,
            expected_base_hash=base_candidate_sha,
            audit=audit,
        )
        trial = {
            "seed": seed,
            "arm": arm,
            "valid": valid,
            "invalid_reasons": reasons,
            "pre_dev_loss": pre_loss,
            "post_dev_loss": post_loss,
            "improvement": pre_loss - post_loss,
            "selected": plans[arm].selected,
            "count_signature": plans[arm].count_signature,
            "plan_sha256": plans[arm].manifest()["plan_sha256"],
            "base_candidate_sha256": base_hash_before,
            "mask_cal_selection_sha256": sha256_json(selection_manifest(mask_rows)),
            "one_step_dev_selection_sha256": sha256_json(selection_manifest(dev_rows)),
            "audit": audit,
            "support_overlaps": overlaps if arm == "real_peers_calibrated" else None,
        }
        trials.append(trial)
        (seed_dir / f"trial_{arm}.json").write_text(
            json.dumps(trial, indent=2, sort_keys=True), encoding="utf-8"
        )

    restore_candidates(candidates, base_snapshot)
    return trials


def write_decision_bundle(
    cfg: Mapping[str, Any],
    *,
    out_dir: Path,
    trials: Sequence[Mapping[str, Any]],
    split_audit: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    base_candidate_sha: str,
) -> dict[str, Any]:
    experiment = cfg["experiment"]
    summary = summarize(
        trials,
        seeds=[int(seed) for seed in experiment["seeds"]],
        required_arms=list(experiment["arms"]),
        real_arm=str(experiment["primary_real_arm"]),
        shuffled_arm=str(experiment["primary_shuffled_arm"]),
        layer_control_arm=str(experiment["primary_layer_control_arm"]),
        min_win_fraction=float(experiment.get("min_win_fraction", 0.8)),
    )
    bundle = out_dir / "decision_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_decision_markdown(summary, bundle / "DECISION.md")
    with (bundle / "TRIALS.jsonl").open("w", encoding="utf-8") as handle:
        for trial in sorted(trials, key=lambda row: (int(row["seed"]), str(row["arm"]))):
            handle.write(json.dumps(trial, sort_keys=True) + "\n")
    (bundle / "SPLIT_AUDIT.json").write_text(
        json.dumps(split_audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    config_public = deepcopy(dict(cfg))
    config_public.pop("_config_path", None)
    manifest = {
        "experiment": config_public["experiment"],
        "config_sha256": sha256_json(config_public),
        "checkpoint": checkpoint,
        "base_candidate_sha256": base_candidate_sha,
        "git": git_state(),
        "environment": environment_manifest(),
        "completed_at_unix": time.time(),
        "claim_boundary": summary["claim_boundary"],
    }
    (bundle / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    plan_manifests: list[dict[str, Any]] = []
    generator_manifests: list[dict[str, Any]] = []
    for seed in cfg["experiment"]["seeds"]:
        seed_dir = out_dir / f"seed_{int(seed)}"
        for path in sorted((seed_dir / "plans").glob("*.manifest.json")):
            plan_manifests.append(json.loads(path.read_text(encoding="utf-8")))
        gpath = seed_dir / "generators" / "manifest.json"
        if gpath.exists():
            row = json.loads(gpath.read_text(encoding="utf-8"))
            row["seed"] = int(seed)
            generator_manifests.append(row)
    with (bundle / "PLAN_MANIFESTS.jsonl").open("w", encoding="utf-8") as handle:
        for row in plan_manifests:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (bundle / "GENERATOR_MANIFESTS.jsonl").open("w", encoding="utf-8") as handle:
        for row in generator_manifests:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    hashes = hash_evidence_tree(bundle)
    (bundle / "ARTIFACT_HASHES.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def run_experiment(cfg: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    indexed, split_audit, out_dir = prepare(cfg)
    device = resolve_device(cfg)
    seed_everything(
        int(cfg["experiment"]["seeds"][0]),
        strict=bool(cfg["model"].get("strict_determinism", True)),
    )
    checkpoint = checkpoint_ledger(
        REPO / cfg["model"]["path"],
        hash_weights=bool(cfg["model"].get("hash_weights", True)),
    )
    tokenizer, model = load_model_and_tokenizer(cfg, device)
    candidates = candidate_parameters(model, str(cfg["pcsm"]["module_path"]))
    base_snapshot = snapshot_candidates(candidates)
    base_candidate_sha = candidate_hash(candidates)

    trials: list[dict[str, Any]] = []
    for seed_raw in cfg["experiment"]["seeds"]:
        seed = int(seed_raw)
        seed_dir = out_dir / f"seed_{seed}"
        complete_path = seed_dir / "complete.json"
        if resume and complete_path.exists():
            payload = json.loads(complete_path.read_text(encoding="utf-8"))
            trial_paths = sorted(seed_dir.glob("trial_*.json"))
            seed_trials = [json.loads(path.read_text(encoding="utf-8")) for path in trial_paths]
            if len(seed_trials) == len(cfg["experiment"]["arms"]) and payload.get("complete"):
                trials.extend(seed_trials)
                continue
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_trials = run_seed(
            cfg=cfg,
            seed=seed,
            indexed=indexed,
            tokenizer=tokenizer,
            model=model,
            candidates=candidates,
            base_snapshot=base_snapshot,
            base_candidate_sha=base_candidate_sha,
            device=device,
            out_dir=out_dir,
        )
        trials.extend(seed_trials)
        complete_path.write_text(
            json.dumps({"complete": True, "seed": seed}, indent=2), encoding="utf-8"
        )

    summary = write_decision_bundle(
        cfg,
        out_dir=out_dir,
        trials=trials,
        split_audit=split_audit,
        checkpoint=checkpoint,
        base_candidate_sha=base_candidate_sha,
    )
    print(json.dumps(summary, indent=2))
    return summary


def verify(cfg: Mapping[str, Any]) -> dict[str, Any]:
    out_dir = REPO / cfg["output"]["dir"]
    bundle = out_dir / "decision_bundle"
    hashes_path = bundle / "ARTIFACT_HASHES.json"
    if not hashes_path.exists():
        raise FileNotFoundError("decision bundle is incomplete: ARTIFACT_HASHES.json missing")
    expected = json.loads(hashes_path.read_text(encoding="utf-8"))
    mismatches = []
    for relative, digest in expected.items():
        path = bundle / relative
        if not path.exists():
            mismatches.append({"path": relative, "reason": "missing"})
        elif sha256_file(path) != digest:
            mismatches.append({"path": relative, "reason": "hash_mismatch"})
    summary = json.loads((bundle / "SUMMARY.json").read_text(encoding="utf-8"))
    trials = [
        json.loads(line)
        for line in (bundle / "TRIALS.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_trials = len(cfg["experiment"]["seeds"]) * len(cfg["experiment"]["arms"])
    if len(trials) != expected_trials:
        mismatches.append(
            {"path": "TRIALS.jsonl", "reason": f"expected {expected_trials} rows, got {len(trials)}"}
        )
    if mismatches:
        raise RuntimeError(f"evidence verification failed: {mismatches}")
    result = {"verified": True, "status": summary["status"], "trial_count": len(trials)}
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run", "verify"):
        p = sub.add_parser(command)
        p.add_argument("--config", required=True)
        if command == "run":
            p.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "prepare":
        _, audit, out_dir = prepare(cfg)
        print(json.dumps({"output_dir": str(out_dir), **audit["split_audit"]}, indent=2))
    elif args.command == "run":
        run_experiment(cfg, resume=bool(args.resume))
    elif args.command == "verify":
        verify(cfg)


if __name__ == "__main__":
    main()
