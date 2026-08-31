#!/usr/bin/env python3
"""Preflight doctor for CRCC v1 discovery execution.

The doctor performs no benchmark or recovery evaluation. It verifies the frozen
scientific contract, repository ancestry, diagnostic assets, pinned checkpoint,
runtime capacity, tokenizer/model compatibility, and the one-layer bypass hook.
It writes a machine-readable diagnostic report under runs/critical_core/.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.critical_core.common import (
    ContractError,
    assert_unsealed_path,
    atomic_write_json,
    checkpoint_inventory,
    load_diagnostic_records,
    load_yaml,
    object_sha256,
    resolve_repo_path,
    sha256_file,
)

PREREG_HEAD = "242613d1b4b5e0a5a4a7154c60337115ca394bd7"
EXPECTED_REPO = "Qwen/Qwen3-0.6B"
EXPECTED_REVISION = "c916fa4defd319b7d4e4da17604ca7338f4d99f5"
EXPECTED_LAYERS = 28
EXPECTED_SOURCES = [
    (
        "rag_train",
        "Aegis_RAG_v3_train_4410/rag_v3_train.sft.jsonl",
        16,
    ),
    (
        "short_preservation_train",
        "Rineya_Short_Response_Behavior_Corpus_v1.0/data/sft/rineya_short_response_preservation_slice.jsonl",
        16,
    ),
]
EXPECTED_THRESHOLDS = {
    "composite_score": 0.60,
    "hard_nll_ratio": 4.0,
    "hard_min_collapse_rate": 0.25,
    "hard_intelligibility_drop": 0.60,
    "hard_nonfinite_nll_rate": 0.0,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def check_contract(config_path: Path) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    model = cfg.get("model", {})
    require(model.get("source_repo") == EXPECTED_REPO, "model.source_repo drifted")
    require(model.get("revision") == EXPECTED_REVISION, "model.revision drifted")
    require(int(model.get("num_hidden_layers", -1)) == EXPECTED_LAYERS, "model depth drifted")

    require(int(cfg.get("seed", -1)) == 20260831, "discovery seed drifted")
    require(str(cfg.get("scan", {}).get("layers")) == "all", "scan.layers must remain all")
    require(int(cfg.get("data", {}).get("minimum_cases", -1)) == 32, "minimum_cases drifted")
    require(int(cfg.get("data", {}).get("max_total_tokens", -1)) == 768, "token cap drifted")
    require(int(cfg.get("generation", {}).get("max_new_tokens", -1)) == 48, "generation cap drifted")
    require(not bool(cfg.get("generation", {}).get("do_sample", True)), "discovery must remain greedy")
    require(not bool(cfg.get("generation", {}).get("enable_thinking", True)), "thinking must remain disabled")
    require(int(cfg.get("criticality", {}).get("minimum_supporting_sources", -1)) == 2, "cross-source support drifted")
    require(cfg.get("criticality", {}).get("manual_hypothesis_layers", []) == [], "manual hypotheses must remain empty")
    require(dict(cfg.get("criticality", {}).get("thresholds", {})) == EXPECTED_THRESHOLDS, "criticality thresholds drifted")

    sources = cfg.get("data", {}).get("sources", [])
    normalized = [(str(x.get("label")), str(x.get("path")), int(x.get("take", -1))) for x in sources]
    require(normalized == EXPECTED_SOURCES, "diagnostic source contract drifted")
    for _, raw_path, _ in EXPECTED_SOURCES:
        path = resolve_repo_path(raw_path, must_exist=True)
        assert_unsealed_path(path, purpose="CRCC runtime doctor")

    records, provenance = load_diagnostic_records(cfg)
    require(len(records) == 32, f"expected 32 deterministic diagnostics, found {len(records)}")
    return {
        "config_sha256": sha256_file(config_path),
        "diagnostic_case_ids": [str(row["id"]) for row in records],
        "diagnostic_case_ids_sha256": object_sha256([str(row["id"]) for row in records]),
        "sources": provenance,
    }


def ensure_model(cfg: dict[str, Any], *, allow_download: bool) -> Path:
    model_path = resolve_repo_path(str(cfg["model"]["path"]))
    if (model_path / "config.json").is_file():
        return model_path
    if not allow_download:
        raise ContractError(
            f"pinned checkpoint is absent at {model_path}; rerun without --offline to materialize it"
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ContractError("huggingface_hub is required to download the pinned checkpoint") from exc
    model_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=EXPECTED_REPO,
        revision=EXPECTED_REVISION,
        local_dir=str(model_path),
    )
    return model_path


def runtime_inventory(torch_module=None) -> dict[str, Any]:
    disk = shutil.disk_usage(ROOT)
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "disk_free_gib": round(disk.free / (1024**3), 3),
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        report["ram_available_gib"] = round(memory.available / (1024**3), 3)
        report["ram_total_gib"] = round(memory.total / (1024**3), 3)
    except ImportError:
        report["ram_available_gib"] = None
        report["ram_total_gib"] = None

    if torch_module is not None:
        torch = torch_module
        report["torch"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            report["cuda_device"] = props.name
            report["cuda_total_gib"] = round(total_bytes / (1024**3), 3)
            report["cuda_free_gib"] = round(free_bytes / (1024**3), 3)
    return report


def model_smoke(model_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from scripts.critical_core.scan_layer_ablation import _bypass_layer, _decoder_layers

    requested_device = str(cfg.get("device", "auto")).lower()
    use_cuda = torch.cuda.is_available() and requested_device != "cpu"
    device = torch.device("cuda:0" if use_cuda else "cpu")
    requested_dtype_name = str(cfg.get("model", {}).get("dtype", "bfloat16"))
    dtype_table = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    require(requested_dtype_name in dtype_table, f"unsupported configured dtype {requested_dtype_name}")
    load_dtype = dtype_table[requested_dtype_name] if use_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=load_dtype,
        attn_implementation=cfg.get("model", {}).get("attn_implementation", "eager"),
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.config.use_cache = False
    model.to(device).eval()
    layers = _decoder_layers(model)
    require(len(layers) == EXPECTED_LAYERS, f"loaded model exposes {len(layers)} decoder layers")

    text = "CRCC runtime compatibility smoke."
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    require(ids.numel() >= 2, "tokenizer smoke text produced too few tokens")
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        intact = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        with _bypass_layer(layers, 0):
            bypass = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    require(torch.isfinite(intact).all().item(), "intact smoke logits are non-finite")
    require(torch.isfinite(bypass).all().item(), "bypass smoke logits are non-finite")
    require(tuple(intact.shape) == tuple(bypass.shape), "bypass changed output tensor shape")

    return {
        "device": str(device),
        "resolved_dtype": str(load_dtype).replace("torch.", ""),
        "transformers": transformers.__version__,
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_class": model.__class__.__name__,
        "decoder_layers": len(layers),
        "smoke_tokens": int(ids.shape[-1]),
        "intact_finite": True,
        "layer_00_bypass_finite": True,
        "shape": list(intact.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/critical_core_scan_v1.yaml")
    parser.add_argument("--report", default="runs/critical_core/crcc-v1/diagnostics/runtime_doctor.json")
    parser.add_argument("--offline", action="store_true", help="refuse checkpoint download")
    parser.add_argument("--model-smoke", action="store_true", help="load model and test intact + bypass forward compatibility")
    args = parser.parse_args()

    report_path = resolve_repo_path(args.report)
    report: dict[str, Any] = {
        "schema_version": "crcc.runtime-doctor.v1",
        "started_at": utc_now(),
        "status": "RUNNING",
        "preregistration_head": PREREG_HEAD,
        "scientific_evidence": False,
        "sealed_evaluation_opened": False,
    }
    try:
        config_path = resolve_repo_path(args.config, must_exist=True)
        cfg = load_yaml(config_path)

        current_head = git_output("rev-parse", "HEAD")
        branch = git_output("branch", "--show-current")
        ancestry_ok = None
        if current_head is not None:
            completed = subprocess.run(
                ["git", "merge-base", "--is-ancestor", PREREG_HEAD, current_head],
                cwd=ROOT,
                check=False,
            )
            ancestry_ok = completed.returncode == 0
            require(ancestry_ok, f"current HEAD {current_head} is not descended from preregistration head")
        report["git"] = {
            "head": current_head,
            "branch": branch,
            "preregistration_is_ancestor": ancestry_ok,
            "worktree_status": git_output("status", "--porcelain"),
        }

        contract = check_contract(config_path)
        report["contract"] = contract

        try:
            import torch
            import transformers

            report["runtime"] = runtime_inventory(torch)
            report["runtime"]["transformers"] = transformers.__version__
        except ImportError as exc:
            raise ContractError("torch and transformers must be installed before CRCC execution") from exc

        model_path = ensure_model(cfg, allow_download=not args.offline)
        inventory = checkpoint_inventory(model_path, hash_weights=True)
        weight_files = [row for row in inventory.get("files", []) if str(row.get("name", "")).endswith((".safetensors", ".bin", ".pt", ".pth"))]
        require(bool(weight_files), "checkpoint inventory contains no recognized weight file")
        require(all(row.get("sha256") for row in weight_files), "checkpoint weights were not hashed")

        config_json = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        require(int(config_json.get("num_hidden_layers", -1)) == EXPECTED_LAYERS, "checkpoint config is not 28 layers")
        report["model"] = {
            "path": str(model_path.relative_to(ROOT) if model_path.is_relative_to(ROOT) else model_path),
            "source_repo": EXPECTED_REPO,
            "revision": EXPECTED_REVISION,
            "inventory": inventory,
        }

        contract_payload = {
            "schema_version": "crcc.discovery-contract.v1",
            "preregistration_head": PREREG_HEAD,
            "config_sha256": contract["config_sha256"],
            "model_source_repo": EXPECTED_REPO,
            "model_revision": EXPECTED_REVISION,
            "model_inventory_sha256": inventory.get("inventory_sha256"),
            "model_depth": EXPECTED_LAYERS,
            "diagnostic_sources": contract["sources"],
            "diagnostic_case_ids": contract["diagnostic_case_ids"],
            "diagnostic_case_ids_sha256": contract["diagnostic_case_ids_sha256"],
            "generation": cfg.get("generation", {}),
            "criticality": cfg.get("criticality", {}),
        }
        contract_payload["contract_sha256"] = object_sha256(contract_payload)
        contract_path = report_path.parent / "discovery_contract.json"
        atomic_write_json(contract_path, contract_payload)
        report["discovery_contract"] = {
            "path": str(contract_path.relative_to(ROOT) if contract_path.is_relative_to(ROOT) else contract_path),
            "contract_sha256": contract_payload["contract_sha256"],
        }

        if args.model_smoke:
            report["model_smoke"] = model_smoke(model_path, cfg)
        else:
            report["model_smoke"] = {"executed": False}

        report["status"] = "PASS"
        report["completed_at"] = utc_now()
        atomic_write_json(report_path, report)
        print("CRCC runtime doctor: PASS")
        print(f"contract: {report['discovery_contract']['contract_sha256']}")
        print(f"report: {report_path}")
    except Exception as exc:
        report["status"] = "FAIL"
        report["completed_at"] = utc_now()
        report["failure"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(report_path, report)
        print(f"CRCC runtime doctor: FAIL — {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
