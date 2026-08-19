"""End-to-end integration over the real dataset with the reference backend."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rineya_lab.analysis.compare import compare_runs, load_run
from rineya_lab.config import load_config
from rineya_lab.evaluation.runner import evaluate
from rineya_lab.training.loop import dry_run

pytestmark = pytest.mark.usefixtures("dataset")

PROBLEM_LIMIT = 12


def eval_cfg(dataset_root, tmp_path, *, name, condition, view, protocols, mode="oracle", noise=0.0):
    return load_config(
        None,
        extra={
            "dataset": {"root": str(dataset_root), "verify_hashes": False},
            "experiment": {"name": name, "condition": condition, "seed": 20260818,
                           "output_dir": str(tmp_path / name)},
            "data_view": {"view": view},
            "evaluation": {
                "split": "dev",
                "limit": PROBLEM_LIMIT,
                "protocols": protocols,
                "audit_sample_size": 4,
                "backend": {"kind": "reference", "reference_mode": mode, "reference_noise": noise},
            },
        },
    )


def test_full_arm_end_to_end(dataset, dataset_root, tmp_path):
    cfg = eval_cfg(dataset_root, tmp_path, name="full", condition="FULL", view="full_context",
                   protocols=["full_direct", "direct_answer"])
    report = evaluate(cfg, dataset)
    assert report["protocols"]["full_direct"]["strict"]["accuracy"] == 1.0
    assert report["protocols"]["full_direct"]["total_items"] == PROBLEM_LIMIT
    records = (Path(cfg.experiment.output_dir) / "records" / "full_direct.jsonl")
    assert records.exists()
    assert (Path(cfg.experiment.output_dir) / "eval_report.md").exists()
    assert report["warning"].startswith("This report was produced with the reference backend")


def test_split_arm_all_protocols(dataset, dataset_root, tmp_path):
    protocols = [
        "oracle_transition", "full_history_transition", "no_state_transition",
        "last_state_only", "shuffled_state", "distractor_state",
        "oracle_trajectory", "generated_rollout",
    ]
    cfg = eval_cfg(dataset_root, tmp_path, name="split", condition="SPLIT_STATEFUL",
                   view="split_stateful", protocols=protocols)
    report = evaluate(cfg, dataset)
    for protocol in protocols:
        assert protocol in report["protocols"], protocol
    assert report["protocols"]["oracle_transition"]["strict"]["accuracy"] == 1.0
    assert report["protocols"]["generated_rollout"]["strict"]["accuracy"] == 1.0
    derived = report["derived"]
    assert derived["free_rollout_gap"]["gap"] == 0.0
    assert derived["state_sufficiency"]["compact_state_accuracy"] == 1.0


def test_every_item_stays_in_the_denominator(dataset, dataset_root, tmp_path):
    """Unparseable output must never shrink the denominator."""
    cfg = eval_cfg(dataset_root, tmp_path, name="noisy", condition="SPLIT_STATEFUL",
                   view="split_stateful", protocols=["oracle_transition"], mode="noisy", noise=0.9)
    report = evaluate(cfg, dataset)
    summary = report["protocols"]["oracle_transition"]
    assert summary["scored_items"] == summary["total_items"]
    distribution = summary["failure_distribution"]
    assert sum(distribution.values()) == summary["total_items"]
    assert set(distribution) - {"correct"}, "noisy mode should produce failures"


def test_perturbations_are_visible_in_the_records(dataset, dataset_root, tmp_path):
    cfg = eval_cfg(dataset_root, tmp_path, name="perturb", condition="SPLIT_STATEFUL",
                   view="split_stateful", protocols=["shuffled_state", "distractor_state"])
    evaluate(cfg, dataset)
    for protocol in ("shuffled_state", "distractor_state"):
        rows = [json.loads(line) for line in
                (Path(cfg.experiment.output_dir) / "records" / f"{protocol}.jsonl").read_text().splitlines()]
        assert rows and all(r["meta"].get("perturbation") == protocol for r in rows)
        # the prompt must actually contain a state that is not the canonical one
        assert any(r["meta"].get("source_problem_id") or r["meta"].get("source_step_index") for r in rows)


def test_paired_comparison_of_two_runs(dataset, dataset_root, tmp_path):
    full_cfg = eval_cfg(dataset_root, tmp_path, name="cmp_full", condition="FULL",
                        view="full_context", protocols=["full_direct"])
    split_cfg = eval_cfg(dataset_root, tmp_path, name="cmp_split", condition="SPLIT_STATEFUL",
                         view="split_stateful", protocols=["generated_rollout"], mode="noisy", noise=0.8)
    evaluate(full_cfg, dataset)
    evaluate(split_cfg, dataset)

    compare_config = load_config(None, extra={
        "comparison": {
            "bootstrap_iterations": 500,
            "randomization_iterations": 500,
            "protocol_pairs": ["full_direct:generated_rollout"],
            "margin_frozen": True,
        }
    })
    report = compare_runs(full_cfg.experiment.output_dir, split_cfg.experiment.output_dir,
                          compare_config, output_dir=tmp_path / "comparison")
    key = "full_direct__vs__generated_rollout"
    assert key in report["comparisons"]
    comparison = report["comparisons"][key]
    assert comparison["n_paired"] == PROBLEM_LIMIT
    assert comparison["bootstrap"]["observed_difference"] < 0   # noisy candidate is worse
    assert comparison["decision"]["superiority"] in ("baseline_superior", "not_established")
    assert (tmp_path / "comparison" / "comparison_report.md").exists()
    assert any("reference backend" in w for w in report["validity_warnings"])


def test_comparison_pairs_on_problem_not_transition(dataset, dataset_root, tmp_path):
    full_cfg = eval_cfg(dataset_root, tmp_path, name="pair_full", condition="FULL",
                        view="full_context", protocols=["full_direct"])
    split_cfg = eval_cfg(dataset_root, tmp_path, name="pair_split", condition="SPLIT_STATEFUL",
                         view="split_stateful", protocols=["oracle_transition"])
    evaluate(full_cfg, dataset)
    evaluate(split_cfg, dataset)
    split_run = load_run(split_cfg.experiment.output_dir)
    # many transition rows...
    assert len(split_run.records_by_protocol["oracle_transition"]) > PROBLEM_LIMIT
    compare_config = load_config(None, extra={"comparison": {
        "bootstrap_iterations": 200, "randomization_iterations": 200,
        "protocol_pairs": ["full_direct:oracle_transition"], "margin_frozen": True}})
    report = compare_runs(full_cfg.experiment.output_dir, split_cfg.experiment.output_dir, compare_config)
    # ...but exactly PROBLEM_LIMIT paired units
    assert report["comparisons"]["full_direct__vs__oracle_transition"]["n_paired"] == PROBLEM_LIMIT


def test_dry_run_accounts_tokens_for_both_arms(dataset, dataset_root, tmp_path):
    for condition, view in (("FULL", "full_context"), ("SPLIT_STATEFUL", "split_stateful")):
        cfg = load_config(None, extra={
            "dataset": {"root": str(dataset_root), "verify_hashes": False, "train_split": "dev"},
            "experiment": {"name": f"dry_{view}", "condition": condition,
                           "output_dir": str(tmp_path / f"dry_{view}")},
            "model": {"name_or_path": "builtin:byte", "tokenizer_name_or_path": "builtin:byte"},
            "data_view": {"view": view, "max_records": 40},
            "training": {"dry_run": True, "max_seq_len": 256, "per_device_batch_size": 2,
                         "gradient_accumulation_steps": 2},
        })
        report = dry_run(cfg, dataset)
        exposure = report["exposure"]
        assert exposure["records_seen"] == 40
        assert exposure["target_tokens"] > 0 and exposure["input_tokens"] > 0
        assert report["template"]["template_source"] == "portable_fallback"


def test_full_targets_are_longer_than_split_targets(dataset, dataset_root, tmp_path):
    """Sanity check on the representation difference the budget controls exist for."""
    totals = {}
    for condition, view in (("FULL", "full_context"), ("SPLIT_STATEFUL", "split_stateful")):
        cfg = load_config(None, extra={
            "dataset": {"root": str(dataset_root), "verify_hashes": False, "train_split": "dev"},
            "experiment": {"name": f"len_{view}", "condition": condition,
                           "output_dir": str(tmp_path / f"len_{view}")},
            "model": {"name_or_path": "builtin:byte", "tokenizer_name_or_path": "builtin:byte"},
            "data_view": {"view": view, "max_records": 100},
            "training": {"dry_run": True, "max_seq_len": 512},
        })
        exposure = dry_run(cfg, dataset)["exposure"]
        totals[view] = exposure["target_tokens"] / exposure["records_seen"]
    assert totals["full_context"] > totals["split_stateful"]


def test_next_step_only_ablation_is_scored_on_its_own_contract(dataset, dataset_root, tmp_path):
    """The next-step adapter emits prose, not JSON; scoring must not penalise that."""
    cfg = load_config(None, extra={
        "dataset": {"root": str(dataset_root), "verify_hashes": False},
        "experiment": {"name": "next_step", "condition": "SPLIT_NEXT_STEP", "seed": 20260818,
                       "output_dir": str(tmp_path / "next_step")},
        "data_view": {"view": "split_next_step"},
        "evaluation": {"split": "dev", "limit": 6, "protocols": ["oracle_transition"],
                       "audit_sample_size": 2,
                       "backend": {"kind": "reference", "reference_mode": "oracle"}},
    })
    report = evaluate(cfg, dataset)
    summary = report["protocols"]["oracle_transition"]
    assert summary["strict"]["accuracy"] == 1.0
    assert summary["extraction_routes"].get("plain_text_step") == summary["total_items"]


def test_next_step_condition_rejects_rollout_protocols(dataset_root):
    from rineya_lab.config import ConfigError

    with pytest.raises(ConfigError, match="never\\s+emits an updated compact state"):
        load_config(None, extra={
            "experiment": {"condition": "SPLIT_NEXT_STEP"},
            "data_view": {"view": "split_next_step"},
            "evaluation": {"protocols": ["oracle_transition", "generated_rollout"]},
        })
