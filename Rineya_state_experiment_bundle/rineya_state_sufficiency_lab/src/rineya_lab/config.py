"""Configuration model for every experiment in the lab.

Design notes
------------
* One nested dataclass tree covers dataset, model, LoRA, data view, training,
  budget, evaluation and comparison. A run records the *fully resolved* config,
  not the file it came from, so a run is reproducible even if the YAML changes.
* YAML files support ``extends: <relative-or-absolute path>`` for single
  inheritance plus deep-merge, which keeps the condition configs small enough to
  diff against each other at a glance -- an important property when the whole
  point of the experiment is that FULL and SPLIT differ in exactly one thing.
* Unknown keys are a hard error. A silently ignored typo in ``lora.r`` would
  invalidate the comparison rather than fail loudly.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin

import yaml

from rineya_lab.utils.hashing import sha256_json

# --------------------------------------------------------------------------------------
# Enumerated vocabularies (validated at load time)
# --------------------------------------------------------------------------------------

CONDITIONS = (
    "FULL",              # A: problem -> complete rationale + answer
    "SPLIT_STATEFUL",    # C (primary): (problem, state) -> {step, updated state, done, answer}
    "SPLIT_NEXT_STEP",   # ablation: (problem, state) -> next step only
    "BASE",              # untouched base checkpoint, no adapter
)

VIEWS = ("full_context", "split_stateful", "split_next_step")

ORDERS = (
    "shuffled",     # condition C: global shuffle across all problems
    "trajectory",   # condition B: steps preserved within each problem
    "reverse",      # ablation: trajectory order reversed within each problem
    "curriculum",   # ablation: all step_index=1 first, then 2, ...
)

BUDGET_MODES = ("native", "matched_updates", "matched_target_tokens", "matched_problems")

LOSS_WEIGHTINGS = ("none", "inverse_transitions")

# How the adapter that gets evaluated is chosen. The rule must be identical across
# arms and declared in the preregistration; "last" is a valid choice, but it has to
# be a choice rather than an accident.
CHECKPOINT_SELECTIONS = ("last", "best_dev_loss")

PRECISIONS = ("auto", "bf16", "fp16", "fp32")

QUANTIZATIONS = ("none", "4bit", "8bit")

PROTOCOLS = (
    "full_direct",              # FULL: problem -> rationale -> extracted answer
    "direct_answer",            # answer-only prompt, no rationale requested
    "oracle_transition",        # gold compact state -> next transition (teacher forcing)
    "full_history_transition",  # complete prior history -> next transition
    "no_state_transition",      # problem only, no state (ablation)
    "last_state_only",          # state without the original problem (ablation)
    "shuffled_state",           # state taken from a different step of the same problem
    "distractor_state",         # state taken from an unrelated problem
    "oracle_trajectory",        # walk canonical states end-to-end, score final answer
    "generated_rollout",        # autonomous: model consumes its own generated states
)


# --------------------------------------------------------------------------------------
# Config sections
# --------------------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    name: str = "unnamed"
    condition: str = "FULL"
    seed: int = 20260818
    output_dir: str = "runs/unnamed"
    notes: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class DatasetConfig:
    """Where the Rineya sequence-independence dataset lives.

    ``root`` may point at an extracted dataset directory. ``bundle_zip`` points
    at the packaged archive; when set and ``root`` is missing, the archive is
    extracted (with zip-slip protection) into ``extract_dir``.
    """

    root: str = "../rineya_sequence_independence_dataset_v1"
    bundle_zip: str | None = None
    extract_dir: str = "data/extracted"
    verify_hashes: bool = True
    require_manifest: bool = True
    train_split: str = "train"
    dev_split: str = "dev"
    test_split: str = "test"


@dataclass
class ModelConfig:
    name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    revision: str | None = None
    tokenizer_name_or_path: str | None = None
    trust_remote_code: bool = False
    precision: str = "auto"
    attn_implementation: str | None = None
    quantization: str = "none"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    device_map: str | None = None
    local_files_only: bool = False
    chat_template_override: str | None = None
    use_portable_template_fallback: bool = True
    # Extra keyword arguments forwarded to tokenizer.apply_chat_template.
    # Qwen3 templates read `enable_thinking` here: leaving it undefined lets the
    # model emit free-form <think> blocks, which changes response length and
    # content. Pin it explicitly so both arms and the base-model baseline are
    # rendered identically.
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoraConfig:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    modules_to_save: list[str] = field(default_factory=list)
    task_type: str = "CAUSAL_LM"


@dataclass
class DataViewConfig:
    """Which representation of the same 48,000 problems the trainer consumes."""

    view: str = "full_context"
    order: str = "shuffled"
    one_transition_per_problem: bool = False
    step_index_min: int | None = None
    step_index_max: int | None = None
    categories: list[str] = field(default_factory=list)
    max_records: int | None = None
    max_unique_problems: int | None = None
    cache_dir: str | None = None
    reshuffle_each_epoch: bool = True


@dataclass
class TrainingConfig:
    epochs: float = 1.0
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    warmup_steps: int | None = None
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    max_seq_len: int = 1024
    prompt_head_keep: int = 64
    gradient_checkpointing: bool = False
    loss_weighting: str = "none"
    logging_steps: int = 10
    save_steps: int = 0
    save_total_limit: int = 3
    eval_steps: int = 0
    resume_from: str | None = None
    dataloader_num_workers: int = 0
    optimizer: str = "adamw_torch"
    label_smoothing: float = 0.0
    dry_run: bool = False
    device: str | None = None
    # Keep LoRA parameters in fp32 even when the frozen base is bf16/fp16. This is
    # the standard mixed-precision recipe: without it, fp16 adapter weights lose
    # small updates to rounding and the two arms can diverge for reasons that have
    # nothing to do with the training representation.
    fp32_trainable_params: bool = True
    checkpoint_selection: str = "last"
    dev_eval_batch_size: int = 8
    dev_eval_max_records: int = 512


@dataclass
class BudgetConfig:
    """Budget matching. See docs/RUNBOOK.md section "Budget regimes".

    ``native`` runs each representation for its own natural epoch. The matched
    modes exist because the SPLIT dataset has ~3.96 rows per source problem: a
    difference produced purely by extra rows is not evidence about the
    hypothesis.
    """

    mode: str = "native"
    max_optimizer_updates: int | None = None
    max_target_tokens: int | None = None
    max_total_tokens: int | None = None
    max_unique_problems: int | None = None
    max_records: int | None = None
    stop_on_first_limit: bool = True


@dataclass
class GenerationConfig:
    max_new_tokens: int = 320
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = False
    repetition_penalty: float = 1.0
    num_return_sequences: int = 1
    stop_strings: list[str] = field(default_factory=list)
    batch_size: int = 8
    seed: int | None = None


@dataclass
class RolloutConfig:
    """Controls for autonomous generated-state rollouts."""

    max_steps_absolute: int = 16
    max_steps_margin: int = 3
    # Off by default: deriving the step budget from the gold trajectory length
    # hands the SPLIT arm a per-problem hint that full_direct never gets. See
    # rineya_lab.evaluation.rollout.max_steps_for. Enable only as a declared
    # secondary analysis.
    use_dataset_step_bound: bool = False
    stagnation_patience: int = 2
    loop_detection: bool = True
    stop_on_invalid: bool = False
    max_invalid_retries: int = 0
    carry_last_valid_state: bool = True


@dataclass
class BackendConfig:
    kind: str = "reference"           # "hf" | "reference"
    adapter_path: str | None = None
    merge_adapter: bool = False
    reference_mode: str = "oracle"    # reference backend behaviour profile
    reference_noise: float = 0.0
    reference_seed: int = 7
    device: str | None = None


@dataclass
class ScoringConfig:
    numeric_tolerance_abs: float = 1e-6
    numeric_tolerance_rel: float = 1e-6
    percent_tolerance_abs: float = 1e-6
    case_sensitive_text_types: list[str] = field(default_factory=lambda: ["string"])
    strip_trailing_punctuation: bool = True
    unit_synonyms: dict[str, str] = field(default_factory=dict)
    answer_type_scorers: dict[str, str] = field(default_factory=dict)
    default_scorer: str = "auto"
    lenient_track_enabled: bool = True
    symbolic_enabled: bool = True
    regex_tasks: dict[str, str] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    protocols: list[str] = field(default_factory=lambda: ["full_direct"])
    split: str = "dev"
    limit: int | None = None
    categories: list[str] = field(default_factory=list)
    shuffle_eval_order: bool = True
    save_raw_generations: bool = True
    save_per_example_records: bool = True
    backend: BackendConfig = field(default_factory=BackendConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    first_error_depth: bool = True
    audit_sample_size: int = 25
    # Decoupled from experiment.seed so that multiple training seeds evaluate the
    # SAME held-out subset. Keyed on experiment.seed, each seed would score a
    # different subset and paired comparisons would shrink to the intersection.
    subset_seed: int = 20260818
    # This model's chat template permits free-form <think>...</think> output.
    # Reasoning scratch must not be mined for answer candidates.
    strip_reasoning_blocks: bool = True


@dataclass
class ComparisonConfig:
    """Statistical decision framework.

    ``non_inferiority_margin_pp`` ships as a placeholder. docs/PREREGISTRATION.md
    requires freezing and justifying the real margin before TEST results are
    opened.
    """

    baseline_run: str | None = None
    candidate_run: str | None = None
    pair_on: str = "problem_id"
    primary_metric: str = "strict_correct"
    # "baseline_protocol:candidate_protocol" pairs. Empty means: compare every
    # protocol present in both runs, plus the default end-to-end pairing below.
    protocol_pairs: list[str] = field(default_factory=list)
    default_end_to_end_pair: str = "full_direct:generated_rollout"
    alpha: float = 0.05
    bootstrap_iterations: int = 10000
    randomization_iterations: int = 10000
    non_inferiority_margin_pp: float = 2.0
    equivalence_margin_pp: float = 2.0
    margin_frozen: bool = False
    margin_justification: str = ""
    stratify_by: list[str] = field(default_factory=lambda: ["category", "total_steps", "failure_type"])
    audit_sample_size: int = 25
    seed: int = 20260818


@dataclass
class LabConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data_view: DataViewConfig = field(default_factory=DataViewConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)

    # ---- serialisation -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        return sha256_json(self.to_dict())

    def validate(self) -> None:
        _validate(self)


class ConfigError(ValueError):
    """Raised for unknown keys, wrong types or invalid enumerated values."""


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml_with_inheritance(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = Path(path).resolve()
    if path in _seen:
        chain = " -> ".join(str(p) for p in (*_seen, path))
        raise ConfigError(f"Circular 'extends' chain: {chain}")
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    parent_ref = raw.pop("extends", None)
    if parent_ref is None:
        return raw
    parent_path = Path(parent_ref)
    if not parent_path.is_absolute():
        parent_path = (path.parent / parent_path).resolve()
    parent = load_yaml_with_inheritance(parent_path, (*_seen, path))
    return _deep_merge(parent, raw)


def _coerce(value: Any, annotation: Any, where: str) -> Any:
    """Coerce a YAML scalar to the dataclass annotation, with strict errors."""
    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)
        # Optional[X] / X | None
        if type(None) in args:
            if value is None:
                return None
            inner = [a for a in args if a is not type(None)]
            return _coerce(value, inner[0], where)
        if origin in (list, tuple):
            if value is None:
                return []
            if not isinstance(value, (list, tuple)):
                raise ConfigError(f"{where}: expected a list, got {type(value).__name__}")
            item_type = args[0] if args else Any
            return [_coerce(item, item_type, f"{where}[]") for item in value]
        if origin is dict:
            if value is None:
                return {}
            if not isinstance(value, dict):
                raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
            key_t, val_t = (args if args else (Any, Any))
            return {_coerce(k, key_t, where): _coerce(v, val_t, where) for k, v in value.items()}
        return value
    if annotation is Any or annotation is None:
        return value
    if annotation is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{where}: expected a boolean, got {value!r}")
    if annotation is int:
        if isinstance(value, bool):
            raise ConfigError(f"{where}: expected an int, got a boolean")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and float(value).is_integer():
            return int(value)
        raise ConfigError(f"{where}: expected an int, got {value!r}")
    if annotation is float:
        if isinstance(value, bool):
            raise ConfigError(f"{where}: expected a float, got a boolean")
        if isinstance(value, (int, float)):
            return float(value)
        raise ConfigError(f"{where}: expected a float, got {value!r}")
    if annotation is str:
        if isinstance(value, str):
            return value
        raise ConfigError(f"{where}: expected a string, got {value!r}")
    return value


def _from_dict(cls: type, data: dict[str, Any], where: str = "") -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{where or cls.__name__}: expected a mapping, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        prefix = f"{where}." if where else ""
        raise ConfigError(
            f"Unknown config key(s): {', '.join(prefix + k for k in unknown)}. "
            f"Valid keys for {cls.__name__}: {', '.join(sorted(known))}"
        )
    kwargs: dict[str, Any] = {}
    for name, f in known.items():
        if name not in data:
            continue
        sub_where = f"{where}.{name}" if where else name
        if is_dataclass(f.type) if isinstance(f.type, type) else False:
            kwargs[name] = _from_dict(f.type, data[name], sub_where)
        else:
            annotation = f.type
            if isinstance(annotation, str):  # postponed annotations
                annotation = _resolve_annotation(annotation)
            if isinstance(annotation, type) and is_dataclass(annotation):
                kwargs[name] = _from_dict(annotation, data[name], sub_where)
            else:
                kwargs[name] = _coerce(data[name], annotation, sub_where)
    return cls(**kwargs)


_ANNOTATION_NAMESPACE: dict[str, Any] = {}


def _resolve_annotation(annotation: str) -> Any:
    if not _ANNOTATION_NAMESPACE:
        import typing

        _ANNOTATION_NAMESPACE.update(vars(typing))
        _ANNOTATION_NAMESPACE.update(globals())
    try:
        return eval(annotation, _ANNOTATION_NAMESPACE)  # noqa: S307 - fixed internal annotations
    except Exception:  # pragma: no cover - defensive
        return Any


def set_by_path(data: dict[str, Any], dotted_key: str, raw_value: str) -> None:
    """Apply a ``--set section.key=value`` override in place (YAML-typed value)."""
    parts = dotted_key.split(".")
    cursor = data
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
        if not isinstance(cursor, dict):
            raise ConfigError(f"--set {dotted_key}: '{part}' is not a section")
    cursor[parts[-1]] = yaml.safe_load(raw_value)


def config_from_dict(data: dict[str, Any]) -> LabConfig:
    cfg = _from_dict(LabConfig, data)
    cfg.validate()
    return cfg


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> LabConfig:
    """Load YAML (with ``extends``), apply dict + dotted overrides, validate."""
    data: dict[str, Any] = load_yaml_with_inheritance(path) if path is not None else {}
    if extra:
        data = _deep_merge(data, extra)
    for override in overrides or []:
        if "=" not in override:
            raise ConfigError(f"--set expects key=value, got {override!r}")
        key, value = override.split("=", 1)
        set_by_path(data, key.strip(), value.strip())
    return config_from_dict(data)


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


def _one_of(value: Any, allowed: tuple[str, ...], where: str) -> None:
    if value not in allowed:
        raise ConfigError(f"{where}: {value!r} is not one of {allowed}")


def _validate(cfg: LabConfig) -> None:
    _one_of(cfg.experiment.condition, CONDITIONS, "experiment.condition")
    _one_of(cfg.data_view.view, VIEWS, "data_view.view")
    _one_of(cfg.data_view.order, ORDERS, "data_view.order")
    _one_of(cfg.budget.mode, BUDGET_MODES, "budget.mode")
    _one_of(cfg.training.loss_weighting, LOSS_WEIGHTINGS, "training.loss_weighting")
    _one_of(cfg.training.checkpoint_selection, CHECKPOINT_SELECTIONS, "training.checkpoint_selection")
    _one_of(cfg.model.precision, PRECISIONS, "model.precision")
    _one_of(cfg.model.quantization, QUANTIZATIONS, "model.quantization")
    _one_of(cfg.evaluation.backend.kind, ("hf", "reference"), "evaluation.backend.kind")

    for protocol in cfg.evaluation.protocols:
        _one_of(protocol, PROTOCOLS, "evaluation.protocols")
    if not cfg.evaluation.protocols:
        raise ConfigError("evaluation.protocols must not be empty")
    if cfg.experiment.condition == "SPLIT_NEXT_STEP":
        impossible = {"generated_rollout", "oracle_trajectory"} & set(cfg.evaluation.protocols)
        if impossible:
            raise ConfigError(
                f"Condition SPLIT_NEXT_STEP cannot run {sorted(impossible)}: the next-step-only adapter never "
                "emits an updated compact state, so there is nothing to continue a rollout from. "
                "Evaluate it with oracle_transition and the state-perturbation protocols instead."
            )

    for split_name, value in (
        ("dataset.train_split", cfg.dataset.train_split),
        ("dataset.dev_split", cfg.dataset.dev_split),
        ("dataset.test_split", cfg.dataset.test_split),
        ("evaluation.split", cfg.evaluation.split),
    ):
        _one_of(value, ("train", "dev", "test"), split_name)

    # Condition / view consistency: the whole experiment turns on this pairing.
    expected_view = {
        "FULL": "full_context",
        "SPLIT_STATEFUL": "split_stateful",
        "SPLIT_NEXT_STEP": "split_next_step",
    }.get(cfg.experiment.condition)
    if expected_view is not None and cfg.data_view.view != expected_view:
        raise ConfigError(
            f"experiment.condition={cfg.experiment.condition} requires data_view.view={expected_view}, "
            f"got {cfg.data_view.view!r}. Mixing views inside one adapter run invalidates the comparison."
        )
    if cfg.experiment.condition == "FULL" and cfg.data_view.order != "shuffled":
        raise ConfigError(
            "The FULL condition has one record per problem; data_view.order must stay 'shuffled'. "
            "Ordering ablations only apply to SPLIT conditions."
        )
    if cfg.experiment.condition == "FULL" and cfg.data_view.one_transition_per_problem:
        raise ConfigError("data_view.one_transition_per_problem is meaningless for the FULL condition")

    if cfg.training.per_device_batch_size < 1:
        raise ConfigError("training.per_device_batch_size must be >= 1")
    if cfg.training.gradient_accumulation_steps < 1:
        raise ConfigError("training.gradient_accumulation_steps must be >= 1")
    if cfg.training.max_seq_len < 32:
        raise ConfigError("training.max_seq_len must be >= 32")
    if not 0 <= cfg.training.prompt_head_keep < cfg.training.max_seq_len:
        raise ConfigError("training.prompt_head_keep must be in [0, training.max_seq_len)")
    if cfg.training.epochs <= 0:
        raise ConfigError("training.epochs must be > 0")
    if cfg.training.checkpoint_selection == "best_dev_loss" and cfg.training.eval_steps < 1:
        raise ConfigError(
            "training.checkpoint_selection=best_dev_loss requires training.eval_steps >= 1 so a DEV "
            "loss is actually measured during the run"
        )
    if cfg.lora.enabled and cfg.lora.r < 1:
        raise ConfigError("lora.r must be >= 1 when lora.enabled is true")
    if cfg.lora.enabled and not cfg.lora.target_modules:
        raise ConfigError("lora.target_modules must be non-empty when lora.enabled is true")

    if cfg.budget.mode == "matched_updates" and cfg.budget.max_optimizer_updates is None:
        raise ConfigError("budget.mode=matched_updates requires budget.max_optimizer_updates")
    if cfg.budget.mode == "matched_target_tokens" and cfg.budget.max_target_tokens is None:
        raise ConfigError("budget.mode=matched_target_tokens requires budget.max_target_tokens")
    if cfg.budget.mode == "matched_problems" and cfg.budget.max_unique_problems is None:
        raise ConfigError("budget.mode=matched_problems requires budget.max_unique_problems")

    if cfg.model.quantization != "none" and not cfg.lora.enabled:
        raise ConfigError("Quantized base weights require lora.enabled=true (QLoRA); full fine-tuning is not supported")

    if not 0 < cfg.comparison.alpha < 1:
        raise ConfigError("comparison.alpha must be in (0, 1)")
    if cfg.comparison.non_inferiority_margin_pp < 0 or cfg.comparison.equivalence_margin_pp < 0:
        raise ConfigError("comparison margins must be non-negative percentage points")
    if cfg.comparison.bootstrap_iterations < 100:
        raise ConfigError("comparison.bootstrap_iterations must be >= 100")
    for pair in [*cfg.comparison.protocol_pairs, cfg.comparison.default_end_to_end_pair]:
        if pair and pair.count(":") != 1:
            raise ConfigError(f"comparison protocol pair {pair!r} must be 'baseline_protocol:candidate_protocol'")
        if pair:
            left, right = pair.split(":")
            for side in (left, right):
                _one_of(side, PROTOCOLS, "comparison.protocol_pairs")

    gen = cfg.evaluation.generation
    if gen.do_sample and gen.temperature <= 0:
        raise ConfigError("evaluation.generation.temperature must be > 0 when do_sample is true")
    if not gen.do_sample and gen.temperature not in (0.0, 1.0):
        raise ConfigError(
            "evaluation.generation.temperature is ignored for greedy decoding; set do_sample=true or temperature=0.0"
        )
    if cfg.evaluation.rollout.max_steps_absolute < 2:
        raise ConfigError("evaluation.rollout.max_steps_absolute must be >= 2")
