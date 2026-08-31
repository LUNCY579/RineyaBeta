"""Strict configuration and experimental-arm contracts for DAST."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

ArmName = Literal[
    "cot_baseline",
    "state_ordered",
    "state_shuffled",
    "state_reverse",
    "state_final_repeated",
    "literal_ordered",
    "operator_ordered",
    "operator_shuffled",
]


@dataclass(frozen=True)
class ArmSpec:
    name: ArmName
    target_kind: Literal["none", "compact_state", "reasoning_step"]
    assignment: Literal["none", "ordered", "shuffled", "reverse", "final_repeated"]
    uses_operators: bool


ARM_SPECS: dict[str, ArmSpec] = {
    "cot_baseline": ArmSpec("cot_baseline", "none", "none", False),
    "state_ordered": ArmSpec("state_ordered", "compact_state", "ordered", False),
    "state_shuffled": ArmSpec("state_shuffled", "compact_state", "shuffled", False),
    "state_reverse": ArmSpec("state_reverse", "compact_state", "reverse", False),
    "state_final_repeated": ArmSpec("state_final_repeated", "compact_state", "final_repeated", False),
    "literal_ordered": ArmSpec("literal_ordered", "reasoning_step", "ordered", False),
    "operator_ordered": ArmSpec("operator_ordered", "compact_state", "ordered", True),
    "operator_shuffled": ArmSpec("operator_shuffled", "compact_state", "shuffled", True),
}


@dataclass
class ExperimentConfig:
    name: str = "dast-dev"
    arm: ArmName = "state_ordered"
    seed: int = 17
    output_dir: str = "runs/dast-dev"
    preregistration_frozen: bool = False
    preregistration_sha256: str = ""
    allow_sealed_test: bool = False
    # Which tier of the EVIDENCE_STANDARD vocabulary this run is allowed to claim.
    # Defaults to the weakest: a run is ENGINEERING unless someone deliberately
    # says otherwise. Previously the manifest status was derived only from
    # eval_split, so a 16-problem seam rehearsal was recorded as DEV_COMPLETE,
    # indistinguishable from a qualifying arm run.
    evidence_status: Literal["engineering", "exploratory_dev", "confirmatory_dev"] = "engineering"


@dataclass
class ModelConfig:
    name_or_path: str = "Qwen/Qwen3-0.6B"
    revision: str = "main"
    local_files_only: bool = True
    trust_remote_code: bool = False
    precision: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    quantization: Literal["none", "4bit", "8bit"] = "none"
    device_map: str | None = None
    attn_implementation: str | None = None
    max_length: int = 512
    checkpoint_fractions: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0)
    # "raw" reads hidden_states[d_k] as committed. On a real HF model the LAST
    # entry is post-final-norm while the others are raw residual states, so the
    # single shared projector receives two representation classes. "final_norm"
    # applies the model's own final norm to every checkpoint that is not already
    # normed, putting all depths in one basis. Defaults to "raw" to preserve the
    # committed behaviour; the shipped configs select "final_norm".
    checkpoint_readout: Literal["raw", "final_norm"] = "raw"
    pooling: Literal["mean", "last_prompt"] = "mean"
    feedback_enabled: bool = False
    feedback_scale: float = 0.05


@dataclass
class DataConfig:
    path: str = ""
    # The reference corpus ships one content file per split under data/full_context/,
    # while `path` assumes a single file carrying every split. These optional
    # per-split overrides let a run read the real corpus without concatenating it,
    # and they keep the sealed split in a file no DEV config ever names.
    train_path: str = ""
    eval_path: str = ""
    train_split: str = "train"
    eval_split: str = "dev"
    max_train_records: int | None = 4000
    max_eval_records: int | None = 800
    num_workers: int = 0
    system_prompt: str = (
        "Solve the problem carefully. Give a concise reasoning trace and make the final answer explicit."
    )
    truncation: Literal["error", "preserve_target"] = "preserve_target"


@dataclass
class SemanticConfig:
    state_dim: int = 128
    projector_hidden_dim: int = 256
    target_projection_seed: int = 1701
    target_max_tokens: int = 64
    # "within_problem" subtracts each problem's canonical-state mean before
    # renormalising. Defaults to "none" so the committed V1 behaviour is
    # unchanged; the shipped configs select the corrected geometry explicitly.
    target_centering: Literal["none", "within_problem"] = "none"
    state_weight: float = 1.0
    contrastive_weight: float = 0.10
    contrastive_temperature: float = 0.07
    stage_weights: tuple[float, ...] = ()


@dataclass
class OperatorConfig:
    enabled: bool = False
    num_operators: int = 8
    rank: int = 8
    router_temperature: float = 1.0
    transition_weight: float = 0.50
    consistency_weight: float = 0.10
    diversity_weight: float = 0.001
    balance_weight: float = 0.001
    frozen_random: bool = False


@dataclass
class LoraConfig:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 1
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2.0e-4
    auxiliary_learning_rate: float | None = None
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    fp32_trainable_params: bool = True
    freeze_base_without_lora: bool = True
    log_every_steps: int = 10
    eval_every_steps: int = 0
    save_every_steps: int = 0
    dataloader_shuffle: bool = True
    deterministic_algorithms: bool = False


@dataclass
class EvaluationConfig:
    max_new_tokens: int = 128
    do_generation: bool = True
    generation_batch_size: int = 4
    temperature: float = 0.0
    top_p: float = 1.0
    bootstrap_samples: int = 5000


@dataclass
class DASTConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    operators: OperatorConfig = field(default_factory=OperatorConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    source_path: str = field(default="", repr=False)

    @property
    def arm_spec(self) -> ArmSpec:
        return ARM_SPECS[self.experiment.arm]

    @property
    def num_stages(self) -> int:
        return len(self.model.checkpoint_fractions)

    def resolved_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute() or not self.source_path:
            return path
        return (Path(self.source_path).parent / path).resolve()

    def data_path_for_split(self, split: str) -> Path:
        """Resolve the corpus file that supplies ``split``.

        Falls back to ``data.path`` so a single-file corpus (and the offline
        fixture) keeps working unchanged.
        """

        name = split.lower()
        if name == self.data.train_split.lower() and self.data.train_path:
            return self.resolved_path(self.data.train_path)
        if name == self.data.eval_split.lower() and self.data.eval_path:
            return self.resolved_path(self.data.eval_path)
        if not self.data.path:
            raise ValueError(
                f"no corpus path configured for split {split!r}: set data.path, "
                "or data.train_path/data.eval_path"
            )
        return self.resolved_path(self.data.path)

    def as_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result.pop("source_path", None)
        return result

    def canonical_sha256(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate(self) -> None:
        errors: list[str] = []
        if self.experiment.arm not in ARM_SPECS:
            errors.append(f"unknown experiment arm: {self.experiment.arm}")
        if self.experiment.seed < 0:
            errors.append("experiment.seed must be non-negative")

        # Literal annotations are not enforced at runtime and load_config only
        # rejects unknown KEYS, so a typo in one of these reached a bare KeyError
        # or a silent wrong branch after the model and corpus were already loaded.
        if self.experiment.evidence_status not in ("engineering", "exploratory_dev", "confirmatory_dev"):
            errors.append(
                "experiment.evidence_status must be engineering, exploratory_dev or "
                f"confirmatory_dev; got {self.experiment.evidence_status!r}"
            )
        if self.model.checkpoint_readout not in ("raw", "final_norm"):
            errors.append(
                f"model.checkpoint_readout must be raw or final_norm; got {self.model.checkpoint_readout!r}"
            )
        if self.semantic.target_centering not in ("none", "within_problem"):
            errors.append(
                "semantic.target_centering must be none or within_problem; "
                f"got {self.semantic.target_centering!r}"
            )
        # A confirmatory claim requires the frozen protocol it is confirming.
        if self.experiment.evidence_status == "confirmatory_dev" and not self.experiment.preregistration_frozen:
            errors.append(
                "experiment.evidence_status='confirmatory_dev' requires "
                "preregistration_frozen: true"
            )

        fractions = self.model.checkpoint_fractions
        if len(fractions) < 2:
            errors.append("at least two checkpoint_fractions are required")
        if any(not (0.0 < value <= 1.0) for value in fractions):
            errors.append("checkpoint_fractions must lie in (0, 1]")
        if tuple(sorted(set(fractions))) != fractions:
            errors.append("checkpoint_fractions must be strictly increasing and unique")
        # The final checkpoint must sit near the top of the stack, but it may not
        # have to BE the top. On a real HF model hidden_states[L] is
        # post-final-norm while every earlier entry is a raw residual state, so a
        # schedule pinned to exactly 1.0 forces two representation classes into
        # the single shared projector. Allowing the last 10% of depth lets a run
        # read layer L-1 instead and keep one basis. The floor still prevents a
        # schedule that quietly stops reasoning supervision half way up.
        if fractions and not (0.9 <= fractions[-1] <= 1.0):
            errors.append(
                "the final checkpoint fraction must lie in [0.9, 1.0]; "
                f"got {fractions[-1]}"
            )
        if fractions and abs(fractions[-1] - 1.0) <= 1e-9 and self.model.checkpoint_readout == "final_norm":
            errors.append(
                "checkpoint_readout='final_norm' with a final fraction of exactly 1.0 leaves "
                "the last checkpoint post-final-norm while the others are re-normed raw "
                "residuals, which is the heterogeneous interface this option exists to fix; "
                "move the last fraction just below 1.0 (e.g. (L-1)/L) or use readout='raw'"
            )
        if self.model.max_length < 32:
            errors.append("model.max_length must be at least 32")
        if self.model.feedback_enabled and self.training.gradient_checkpointing:
            errors.append("feedback_enabled is intentionally incompatible with gradient_checkpointing")
        if not (0.0 <= self.model.feedback_scale <= 1.0):
            errors.append("model.feedback_scale must lie in [0, 1]")

        if self.semantic.state_dim < 2:
            errors.append("semantic.state_dim must be at least 2")
        if self.semantic.projector_hidden_dim < self.semantic.state_dim:
            errors.append("semantic.projector_hidden_dim must be >= semantic.state_dim")
        if self.semantic.contrastive_temperature <= 0:
            errors.append("semantic.contrastive_temperature must be positive")
        if self.semantic.stage_weights and len(self.semantic.stage_weights) != len(fractions):
            errors.append("semantic.stage_weights must be empty or match checkpoint_fractions")
        if any(weight < 0 for weight in self.semantic.stage_weights):
            errors.append("semantic.stage_weights cannot be negative")

        spec = self.arm_spec if self.experiment.arm in ARM_SPECS else None
        auxiliary_weight = self.semantic.state_weight + self.semantic.contrastive_weight
        if spec is not None and spec.target_kind == "none" and auxiliary_weight != 0:
            errors.append("cot_baseline must set state_weight and contrastive_weight to zero")
        if spec is not None and spec.target_kind != "none" and self.semantic.state_weight <= 0:
            errors.append("semantic arms require semantic.state_weight > 0")
        if spec is not None and spec.uses_operators != self.operators.enabled:
            errors.append(
                f"arm {spec.name} requires operators.enabled={spec.uses_operators}; got {self.operators.enabled}"
            )
        if self.operators.enabled:
            if self.operators.num_operators < 2:
                errors.append("operator arms require at least two operators")
            if not (1 <= self.operators.rank <= self.semantic.state_dim):
                errors.append("operators.rank must lie in [1, semantic.state_dim]")
            if self.operators.router_temperature <= 0:
                errors.append("operators.router_temperature must be positive")

        if self.training.epochs < 1 or self.training.batch_size < 1:
            errors.append("training.epochs and training.batch_size must be positive")
        if self.training.gradient_accumulation_steps < 1:
            errors.append("training.gradient_accumulation_steps must be positive")
        if self.training.learning_rate <= 0:
            errors.append("training.learning_rate must be positive")
        if not (0.0 <= self.training.warmup_ratio < 1.0):
            errors.append("training.warmup_ratio must lie in [0, 1)")

        if self.experiment.allow_sealed_test:
            frozen_hash = self.experiment.preregistration_sha256.lower()
            if not self.experiment.preregistration_frozen:
                errors.append("allow_sealed_test requires preregistration_frozen=true")
            if not re.fullmatch(r"[0-9a-f]{64}", frozen_hash):
                errors.append("allow_sealed_test requires a 64-character preregistration_sha256")

        if errors:
            raise ValueError("Invalid DAST configuration:\n- " + "\n- ".join(errors))


_SECTION_TYPES: dict[str, type[Any]] = {
    "experiment": ExperimentConfig,
    "model": ModelConfig,
    "data": DataConfig,
    "semantic": SemanticConfig,
    "operators": OperatorConfig,
    "lora": LoraConfig,
    "training": TrainingConfig,
    "evaluation": EvaluationConfig,
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml_with_extends(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved))
        raise ValueError(f"configuration extends cycle: {chain}")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"configuration root must be a mapping: {resolved}")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    if not isinstance(parent, str):
        raise TypeError("extends must be a relative or absolute path string")
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    return _deep_merge(_load_yaml_with_extends(parent_path, (*stack, resolved)), raw)


def _construct_section(section_name: str, cls: type[Any], raw: Any) -> Any:
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise TypeError(f"{section_name} must be a mapping")
    fields = {item.name: item for item in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise ValueError(f"unknown keys in {section_name}: {', '.join(unknown)}")
    values: dict[str, Any] = {}
    for key, value in raw.items():
        default_value = getattr(cls(), key)
        if isinstance(default_value, tuple) and isinstance(value, list):
            value = tuple(value)
        values[key] = value
    return cls(**values)


def load_config(path: str | Path) -> DASTConfig:
    config_path = Path(path).expanduser().resolve()
    raw = _load_yaml_with_extends(config_path)
    unknown_sections = sorted(set(raw) - set(_SECTION_TYPES))
    if unknown_sections:
        raise ValueError(f"unknown top-level configuration sections: {', '.join(unknown_sections)}")
    kwargs = {
        name: _construct_section(name, cls, raw.get(name))
        for name, cls in _SECTION_TYPES.items()
    }
    config = DASTConfig(**kwargs, source_path=str(config_path))
    config.validate()
    return config


def assert_split_allowed(config: DASTConfig, split: str) -> None:
    """Keep the sealed TEST split closed unless a frozen protocol explicitly opens it."""

    if split.lower() != "test":
        return
    if not config.experiment.allow_sealed_test:
        raise PermissionError("TEST is sealed: experiment.allow_sealed_test is false")
    if not config.experiment.preregistration_frozen:
        raise PermissionError("TEST is sealed: preregistration_frozen is false")
    digest = config.experiment.preregistration_sha256.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise PermissionError("TEST is sealed: preregistration_sha256 is not frozen")
