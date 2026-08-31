#!/usr/bin/env python3
"""Verify the local CRCC v1 checkpoint against the official pinned Qwen artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.critical_core.common import ContractError, load_yaml, resolve_repo_path, sha256_file

EXPECTED_REPO = "Qwen/Qwen3-0.6B"
EXPECTED_REVISION = "c916fa4defd319b7d4e4da17604ca7338f4d99f5"
EXPECTED_MODEL_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
EXPECTED_LAYERS = 28


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/critical_core_scan_v1.yaml")
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config, must_exist=True)
    cfg = load_yaml(config_path)
    model_cfg = cfg.get("model", {})
    if model_cfg.get("source_repo") != EXPECTED_REPO:
        raise ContractError("CRCC checkpoint repository provenance drifted")
    if model_cfg.get("revision") != EXPECTED_REVISION:
        raise ContractError("CRCC checkpoint revision drifted")

    model_dir = resolve_repo_path(str(model_cfg.get("path", "")), must_exist=True)
    config_json_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    if not config_json_path.is_file():
        raise ContractError(f"missing checkpoint config: {config_json_path}")
    if not weights_path.is_file():
        raise ContractError(f"missing official single-file weights: {weights_path}")

    model_json = json.loads(config_json_path.read_text(encoding="utf-8"))
    if int(model_json.get("num_hidden_layers", -1)) != EXPECTED_LAYERS:
        raise ContractError("local checkpoint is not the pinned 28-layer coordinate system")

    observed = sha256_file(weights_path)
    if observed != EXPECTED_MODEL_SHA256:
        raise ContractError(
            "local model.safetensors does not match the pinned official Qwen revision: "
            f"expected {EXPECTED_MODEL_SHA256}, observed {observed}"
        )

    print("CRCC checkpoint verification: PASS")
    print(f"repo: {EXPECTED_REPO}")
    print(f"revision: {EXPECTED_REVISION}")
    print(f"model.safetensors sha256: {observed}")
    print(f"decoder layers: {EXPECTED_LAYERS}")


if __name__ == "__main__":
    main()
