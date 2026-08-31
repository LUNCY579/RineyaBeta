"""Strict configuration and sealed-evaluation guardrails."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .bridges import BridgeSpec
from .donors import DonorSpec
from .graft import GateSpec

SCHEMA_VERSION = 1
FORBIDDEN_DATA_MARKERS = (
    "sealed",
    "final_holdout",
    "private_holdout",
    "hidden_eval",
    "hidden_test",
    "long_context_demo",
    "showcase",
)




def forbidden_data_marker(value: str) -> str | None:
    """Return the rejected marker, treating explicit ``unsealed`` paths as open."""

    lowered = value.lower().replace("-", "_")
    lowered = lowered.replace("unsealed", "open_dev")
    return next((marker for marker in FORBIDDEN_DATA_MARKERS if marker in lowered), None)

@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]
    source: Path
    sha256: str

    @property
    def experiment_id(self) -> str:
        return str(self.raw["experiment_id"])

    @property
    def mode(self) -> str:
        return str(self.raw["mode"])

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def donor_spec(self) -> DonorSpec:
        return DonorSpec.from_mapping(self.raw["graft"]["donor"])

    @property
    def input_bridge_spec(self) -> BridgeSpec:
        graft = self.raw["graft"]
        return BridgeSpec.from_mapping(graft.get("input_bridge") or graft.get("bridge"))

    @property
    def output_bridge_spec(self) -> BridgeSpec:
        graft = self.raw["graft"]
        return BridgeSpec.from_mapping(graft.get("output_bridge") or graft.get("bridge"))

    @property
    def gate_spec(self) -> GateSpec:
        return GateSpec.from_mapping(self.raw["graft"].get("gate"))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_keys(mapping: Mapping[str, Any], required: set[str], where: str) -> None:
    missing = sorted(required - set(mapping))
    if missing:
        raise ValueError(f"missing {where} fields: {missing}")


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown {where} fields: {unknown}")


def _iter_named_strings(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _iter_named_strings(child, (*path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_named_strings(child, (*path, str(index)))
    elif isinstance(value, str):
        yield path, value


def assert_sealed_evaluation_closed(raw: Mapping[str, Any]) -> None:
    policy = raw.get("data_policy", {})
    if policy.get("allow_sealed", False):
        raise ValueError("sealed evaluation is structurally disabled in CAG v1")
    for path, value in _iter_named_strings(raw.get("data", {})):
        marker = forbidden_data_marker(value)
        if marker is not None:
            dotted = ".".join(path)
            raise ValueError(
                f"sealed/hidden data marker {marker!r} rejected at data.{dotted}: {value!r}"
            )
    for path, value in _iter_named_strings(raw.get("evaluation", {})):
        marker = forbidden_data_marker(value)
        if marker is not None:
            dotted = ".".join(path)
            raise ValueError(
                f"sealed/hidden evaluation marker {marker!r} rejected at "
                f"evaluation.{dotted}: {value!r}"
            )


def validate_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("configuration root must be an object")
    root = dict(raw)
    allowed_root = {
        "schema_version",
        "experiment_id",
        "mode",
        "claim_level",
        "seed",
        "host",
        "graft",
        "training",
        "data",
        "evaluation",
        "data_policy",
        "output",
        "memory",
        "notes",
    }
    _reject_unknown(root, allowed_root, "top-level")
    _require_keys(
        root,
        {"schema_version", "experiment_id", "mode", "claim_level", "seed", "host", "graft"},
        "top-level",
    )
    if int(root["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {root['schema_version']}; expected {SCHEMA_VERSION}"
        )
    if root["mode"] not in {"smoke", "plan", "real"}:
        raise ValueError("mode must be one of smoke, plan, real")
    if root["claim_level"] != "infrastructure_only":
        raise ValueError("CAG v1 configs must use claim_level='infrastructure_only'")
    if not str(root["experiment_id"]).strip():
        raise ValueError("experiment_id must be non-empty")
    if int(root["seed"]) < 0:
        raise ValueError("seed must be non-negative")

    host = root["host"]
    if not isinstance(host, Mapping):
        raise ValueError("host must be an object")
    _reject_unknown(
        host,
        {
            "kind",
            "model_path",
            "revision",
            "hidden_size",
            "layer_path",
            "layer_indices",
            "dtype",
            "device",
            "attn_implementation",
            "local_files_only",
        },
        "host",
    )
    _require_keys(host, {"kind", "hidden_size", "layer_indices"}, "host")
    hidden_size = int(host["hidden_size"])
    if hidden_size <= 0:
        raise ValueError("host.hidden_size must be positive")
    layer_indices = host["layer_indices"]
    if not isinstance(layer_indices, list) or not layer_indices:
        raise ValueError("host.layer_indices must be a non-empty list")
    parsed_indices = [int(index) for index in layer_indices]
    if min(parsed_indices) < 0 or len(set(parsed_indices)) != len(parsed_indices):
        raise ValueError("host.layer_indices must contain unique non-negative integers")
    if root["mode"] == "real":
        _require_keys(host, {"model_path"}, "host")
        if not str(host["model_path"]).strip():
            raise ValueError("host.model_path must be non-empty in real mode")

    graft = root["graft"]
    if not isinstance(graft, Mapping):
        raise ValueError("graft must be an object")
    _reject_unknown(
        graft,
        {
            "foreign_dim",
            "donor",
            "bridge",
            "input_bridge",
            "output_bridge",
            "gate",
            "freeze_host",
            "freeze_donor",
            "exact_eval_bypass",
        },
        "graft",
    )
    _require_keys(graft, {"foreign_dim", "donor"}, "graft")
    if int(graft["foreign_dim"]) <= 0:
        raise ValueError("graft.foreign_dim must be positive")
    donor_spec = DonorSpec.from_mapping(graft["donor"])
    if donor_spec.d_model is not None and int(donor_spec.d_model) != int(graft["foreign_dim"]):
        raise ValueError(
            "graft.donor.d_model must equal graft.foreign_dim; hidden width changes "
            "belong in the explicit bridges"
        )
    BridgeSpec.from_mapping(graft.get("input_bridge") or graft.get("bridge"))
    BridgeSpec.from_mapping(graft.get("output_bridge") or graft.get("bridge"))
    GateSpec.from_mapping(graft.get("gate"))
    if graft.get("freeze_host", True) is not True:
        raise ValueError("CAG v1 requires graft.freeze_host=true")
    if graft.get("exact_eval_bypass", True) is not True:
        raise ValueError("CAG v1 requires graft.exact_eval_bypass=true")
    if root["mode"] == "real" and host.get("local_files_only", True) is not True:
        raise ValueError("CAG v1 real mode is local-only")

    data = root.get("data", {})
    if not isinstance(data, Mapping):
        raise ValueError("data must be an object")
    _reject_unknown(
        data,
        {
            "train_mix",
            "dev_mix",
            "split_manifest",
            "cache_dir",
            "max_train_rows",
            "max_dev_rows",
        },
        "data",
    )
    if root["mode"] == "real":
        _require_keys(
            data,
            {"train_mix", "dev_mix", "split_manifest", "cache_dir"},
            "data",
        )

    evaluation = root.get("evaluation", {})
    if not isinstance(evaluation, Mapping):
        raise ValueError("evaluation must be an object")
    _reject_unknown(
        evaluation,
        {
            "dev_batch_size",
            "equivalence_probe_rows",
            "preservation_command",
            "target_command",
            "required_categories",
        },
        "evaluation",
    )
    for key in ("dev_batch_size", "equivalence_probe_rows"):
        if key in evaluation and int(evaluation[key]) <= 0:
            raise ValueError(f"evaluation.{key} must be positive")

    output = root.get("output", {})
    if not isinstance(output, Mapping):
        raise ValueError("output must be an object")
    _reject_unknown(output, {"run_dir", "overwrite"}, "output")

    memory = root.get("memory", {})
    if not isinstance(memory, Mapping):
        raise ValueError("memory must be an object")
    _reject_unknown(memory, {"cap_mib"}, "memory")
    if "cap_mib" in memory and memory["cap_mib"] is not None and int(memory["cap_mib"]) <= 0:
        raise ValueError("memory.cap_mib must be positive or null")

    data_policy = root.get("data_policy", {})
    if not isinstance(data_policy, Mapping):
        raise ValueError("data_policy must be an object")
    _reject_unknown(data_policy, {"allow_sealed", "require_content_hashes"}, "data_policy")
    if data_policy.get("allow_sealed", False) is not False:
        raise ValueError("CAG v1 requires data_policy.allow_sealed=false")
    if data_policy.get("require_content_hashes", False) is not True:
        raise ValueError("CAG v1 requires data_policy.require_content_hashes=true")
    assert_sealed_evaluation_closed(root)

    training = root.get("training", {})
    if not isinstance(training, Mapping):
        raise ValueError("training must be an object")
    _reject_unknown(
        training,
        {
            "steps",
            "lr",
            "weight_decay",
            "grad_clip",
            "batch_size",
            "seq_len",
            "max_len",
            "grad_accum",
            "save_every_steps",
        },
        "training",
    )
    if training:
        for key in ("steps", "batch_size", "seq_len", "max_len", "grad_accum"):
            if key in training and int(training[key]) <= 0:
                raise ValueError(f"training.{key} must be positive")
        if "lr" in training and float(training["lr"]) <= 0:
            raise ValueError("training.lr must be positive")

    return root


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    payload = source.read_bytes()
    raw = json.loads(payload.decode("utf-8"))
    validated = validate_config(raw)
    # Hash the canonical semantic configuration, not incidental whitespace.
    digest = _hash_bytes(canonical_json(validated).encode("utf-8"))
    return ExperimentConfig(raw=validated, source=source, sha256=digest)
