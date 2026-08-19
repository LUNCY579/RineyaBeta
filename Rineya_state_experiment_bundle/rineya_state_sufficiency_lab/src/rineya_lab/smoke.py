"""Offline end-to-end smoke tests.

Runs the entire pipeline -- validation, view planning, dry-run training,
reference-backend evaluation for both arms, and the paired comparison -- without
a GPU, a network connection or a downloaded checkpoint. With torch installed it
additionally trains two real LoRA adapters on a randomly initialised tiny model
and asserts that adapter artifacts appear on disk.

This proves the pipelines are *executable*. It says nothing about the
hypothesis, and every artifact it produces is labelled accordingly.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from rineya_lab.config import LabConfig, load_config
from rineya_lab.utils.logging import get_logger
from rineya_lab.version import __version__

LOGGER = get_logger("smoke")

SMOKE_PROBLEM_LIMIT = 24
SMOKE_RECORD_LIMIT = 64


def _base_overrides(dataset_root: str | None, output_dir: Path) -> dict[str, Any]:
    dataset: dict[str, Any] = {"verify_hashes": False}
    if dataset_root:
        dataset["root"] = dataset_root
    return {
        "dataset": dataset,
        "experiment": {"seed": 20260818, "output_dir": str(output_dir)},
    }


def _full_eval_config(dataset_root: str | None, output_dir: Path, mode: str) -> LabConfig:
    data = _base_overrides(dataset_root, output_dir)
    data["experiment"].update({"name": "smoke_full", "condition": "FULL"})
    data["data_view"] = {"view": "full_context"}
    data["evaluation"] = {
        "split": "dev",
        "limit": SMOKE_PROBLEM_LIMIT,
        "protocols": ["full_direct", "direct_answer"],
        "backend": {"kind": "reference", "reference_mode": mode, "reference_noise": 0.35},
        "audit_sample_size": 5,
    }
    return load_config(None, extra=data)


def _split_eval_config(dataset_root: str | None, output_dir: Path, mode: str) -> LabConfig:
    data = _base_overrides(dataset_root, output_dir)
    data["experiment"].update({"name": "smoke_split", "condition": "SPLIT_STATEFUL"})
    data["data_view"] = {"view": "split_stateful"}
    data["evaluation"] = {
        "split": "dev",
        "limit": SMOKE_PROBLEM_LIMIT,
        "protocols": [
            "oracle_transition",
            "full_history_transition",
            "no_state_transition",
            "last_state_only",
            "shuffled_state",
            "distractor_state",
            "oracle_trajectory",
            "generated_rollout",
        ],
        "backend": {"kind": "reference", "reference_mode": mode, "reference_noise": 0.35},
        "audit_sample_size": 5,
    }
    return load_config(None, extra=data)


def _dry_run_config(dataset_root: str | None, output_dir: Path, condition: str, view: str, order: str) -> LabConfig:
    data = _base_overrides(dataset_root, output_dir)
    data["experiment"].update({"name": f"smoke_dry_{condition.lower()}", "condition": condition})
    data["model"] = {"name_or_path": "builtin:byte", "tokenizer_name_or_path": "builtin:byte"}
    data["data_view"] = {"view": view, "order": order, "max_records": SMOKE_RECORD_LIMIT}
    data["dataset"]["train_split"] = "dev"
    data["training"] = {
        "dry_run": True,
        "epochs": 1,
        "per_device_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "max_seq_len": 256,
        "loss_weighting": "inverse_transitions" if condition != "FULL" else "none",
    }
    return load_config(None, extra=data)


def run_smoke(
    cfg: LabConfig | None = None,
    *,
    dataset_root: str | None = None,
    include_training: bool = False,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Execute the offline smoke suite and return a structured result."""
    from rineya_lab.data.loading import RineyaDataset, resolve_dataset

    started = time.time()
    if dataset_root is None and cfg is not None:
        dataset_root = cfg.dataset.root

    temp_dir: tempfile.TemporaryDirectory | None = None
    if output_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="rineya-smoke-")
        output_dir = Path(temp_dir.name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"lab_version": __version__, "status": "PASS", "steps": {}, "output_dir": str(output_dir)}

    def record(name: str, ok: bool, detail: Any) -> None:
        results["steps"][name] = {"ok": ok, "detail": detail}
        if not ok:
            results["status"] = "FAIL"

    try:
        # 1. dataset resolution + limited validation
        from rineya_lab.config import DatasetConfig
        from rineya_lab.data.validate import validate_dataset

        dataset_cfg = DatasetConfig(root=dataset_root or DatasetConfig().root, verify_hashes=False)
        paths = resolve_dataset(dataset_cfg)
        dataset = RineyaDataset(paths, verify_hashes=False)
        record("dataset_resolution", True, paths.as_dict())

        validation = validate_dataset(dataset, splits=("dev",), limit=400)
        record("dataset_validation_sample", validation.status == "PASS",
               {"status": validation.status, "errors": validation.errors[:5], "counts": validation.counts})

        # 2. view plans for all three conditions of the design
        from rineya_lab.data.views import build_view_plan, order_diagnostics

        plans: dict[str, Any] = {}
        for label, condition, view, order in (
            ("A_full", "FULL", "full_context", "shuffled"),
            ("B_ordered_transitions", "SPLIT_STATEFUL", "split_stateful", "trajectory"),
            ("C_unordered_transitions", "SPLIT_STATEFUL", "split_stateful", "shuffled"),
        ):
            plan_cfg = _dry_run_config(dataset_root, output_dir / "plans", condition, view, order)
            plan = build_view_plan(dataset, plan_cfg, split="dev")
            plans[label] = {"plan": plan.summary(), "order": order_diagnostics(plan)}
        adjacency_ok = _order_ablation_is_real(plans)
        record("view_plans", adjacency_ok, plans)

        # 3. dry-run training on both arms (tokenizes and accounts, no model)
        from rineya_lab.training.loop import dry_run

        dry: dict[str, Any] = {}
        for label, condition, view, order in (
            ("full", "FULL", "full_context", "shuffled"),
            ("split", "SPLIT_STATEFUL", "split_stateful", "shuffled"),
        ):
            dry_cfg = _dry_run_config(dataset_root, output_dir / f"dry_{label}", condition, view, order)
            dry[label] = dry_run(dry_cfg, dataset)
        ok = all(entry["exposure"]["target_tokens"] > 0 for entry in dry.values())
        record("dry_run_training", ok, {k: v["exposure"] for k, v in dry.items()})

        # 4. reference-backend evaluation for both arms
        from rineya_lab.evaluation.runner import evaluate

        full_dir = output_dir / "eval_full"
        split_dir = output_dir / "eval_split"
        full_report = evaluate(_full_eval_config(dataset_root, full_dir, "oracle"), dataset)
        split_report = evaluate(_split_eval_config(dataset_root, split_dir, "oracle"), dataset)
        oracle_ok = (
            full_report["protocols"]["full_direct"]["strict"]["accuracy"] == 1.0
            and split_report["protocols"]["oracle_transition"]["strict"]["accuracy"] == 1.0
            and split_report["protocols"]["generated_rollout"]["strict"]["accuracy"] == 1.0
        )
        record(
            "reference_evaluation_oracle",
            oracle_ok,
            {
                "full_direct": full_report["protocols"]["full_direct"]["strict"],
                "oracle_transition": split_report["protocols"]["oracle_transition"]["strict"],
                "generated_rollout": split_report["protocols"]["generated_rollout"]["strict"],
                "free_rollout_gap": split_report["derived"]["free_rollout_gap"],
            },
        )

        # 5. adversarial reference modes must be detected, not silently passed
        detection: dict[str, Any] = {}
        for mode, expected in (
            ("refusal", "refusal"),
            ("empty", "empty_output"),
            ("looping", "loop"),
            ("stagnant", "stagnation"),
            ("premature", "premature_completion"),
            ("nonterminating", "nontermination"),
            ("malformed_json", "correct"),
        ):
            probe_dir = output_dir / f"probe_{mode}"
            probe_cfg = _split_eval_config(dataset_root, probe_dir, mode)
            probe_cfg.evaluation.protocols = ["generated_rollout"]
            probe_cfg.evaluation.limit = 8
            probe_cfg.evaluation.audit_sample_size = 2
            probe = evaluate(probe_cfg, dataset)
            distribution = probe["protocols"]["generated_rollout"]["failure_distribution"]
            detection[mode] = distribution
            if expected is not None and expected not in distribution:
                record(f"failure_detection_{mode}", False, distribution)
        if all(not key.startswith("failure_detection_") for key in results["steps"]):
            record("failure_detection", True, detection)

        # 6. paired comparison across the two runs
        from rineya_lab.analysis.compare import compare_runs

        compare_cfg = _split_eval_config(dataset_root, output_dir / "comparison", "oracle")
        compare_cfg.comparison.bootstrap_iterations = 500
        compare_cfg.comparison.randomization_iterations = 500
        comparison = compare_runs(full_dir, split_dir, compare_cfg, output_dir=output_dir / "comparison")
        record("paired_comparison", bool(comparison["comparisons"]),
               {"pairs": list(comparison["comparisons"]), "warnings": comparison["validity_warnings"]})

        # 7. optional: real tiny-model adapter training
        if include_training:
            record("tiny_model_training", *_tiny_training(dataset, dataset_root, output_dir))

    except Exception as exc:  # pragma: no cover - surfaced as a failing smoke result
        results["status"] = "FAIL"
        results["error"] = f"{type(exc).__name__}: {exc}"
        LOGGER.exception("Smoke test failed")

    results["elapsed_seconds"] = round(time.time() - started, 3)
    results["note"] = (
        "The smoke suite validates that the pipelines execute and that the harness detects known failure "
        "modes. It is not a scientific result about the hypothesis."
    )
    if temp_dir is not None and results["status"] == "PASS":
        temp_dir.cleanup()
    return results


def _order_ablation_is_real(plans: dict[str, Any]) -> bool:
    """Condition B must keep steps adjacent; condition C must not."""
    ordered = plans["B_ordered_transitions"]["order"]["epoch_0"]["adjacent_consecutive_steps"]
    unordered = plans["C_unordered_transitions"]["order"]["epoch_0"]["adjacent_same_problem"]
    return ordered > 0 and unordered == 0


def _tiny_training(dataset, dataset_root: str | None, output_dir: Path) -> tuple[bool, Any]:
    """Train two real LoRA adapters on a randomly initialised tiny model."""
    try:
        import torch  # noqa: F401
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError as exc:
        return True, {"skipped": True, "reason": f"torch/transformers unavailable: {exc}"}

    from rineya_lab.training.loop import train
    from rineya_lab.training.stub_tokenizer import VOCAB_SIZE

    tiny_dir = output_dir / "tiny_model"
    if not (tiny_dir / "config.json").exists():
        config = LlamaConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        model = LlamaForCausalLM(config)
        model.save_pretrained(str(tiny_dir))

    detail: dict[str, Any] = {}
    ok = True
    for label, condition, view in (("full", "FULL", "full_context"), ("split", "SPLIT_STATEFUL", "split_stateful")):
        run_dir = output_dir / f"train_{label}"
        data = _base_overrides(dataset_root, run_dir)
        data["experiment"].update({"name": f"smoke_train_{label}", "condition": condition})
        data["model"] = {"name_or_path": str(tiny_dir), "tokenizer_name_or_path": "builtin:byte", "precision": "fp32"}
        data["dataset"]["train_split"] = "dev"
        data["data_view"] = {"view": view, "max_records": 16}
        data["lora"] = {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]}
        data["training"] = {
            "epochs": 1,
            "per_device_batch_size": 2,
            "gradient_accumulation_steps": 2,
            # 512 byte-tokens fits nearly every DEV target, so the smoke run exercises
            # the normal path rather than the truncation path.
            "max_seq_len": 512,
            "learning_rate": 1e-3,
            "logging_steps": 1,
        }
        cfg = load_config(None, extra=data)
        manifest = train(cfg, dataset)
        adapter_dir = run_dir / "adapter"
        produced = sorted(p.name for p in adapter_dir.iterdir()) if adapter_dir.exists() else []
        has_weights = any(name.startswith("adapter_model") for name in produced)
        ok = ok and has_weights and manifest["training"]["exposure"]["optimizer_updates"] > 0
        detail[label] = {
            "adapter_files": produced,
            "exposure": manifest["training"]["exposure"],
            "stop_reason": manifest["training"]["stop_reason"],
        }
    return ok, detail
