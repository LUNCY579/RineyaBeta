"""Regression tests for the GPU-readiness fixes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rineya_lab.analysis.multiseed import aggregate, aggregate_reports, collect_seed_entries
from rineya_lab.config import ComparisonConfig, load_config
from rineya_lab.evaluation.backends import ReferenceBackend
from rineya_lab.evaluation.protocols import build_eval_set, run_protocol
from rineya_lab.evaluation.runner import evaluate

pytestmark = pytest.mark.usefixtures("dataset")


def make_cfg(dataset_root, tmp_path, *, name, condition, view, protocols, batch_size, seed=1, mode="noisy"):
    return load_config(None, extra={
        "dataset": {"root": str(dataset_root), "verify_hashes": False},
        "experiment": {"name": name, "condition": condition, "seed": seed,
                       "output_dir": str(tmp_path / name)},
        "data_view": {"view": view},
        "evaluation": {"split": "dev", "limit": 10, "protocols": protocols, "audit_sample_size": 2,
                       "generation": {"batch_size": batch_size},
                       "backend": {"kind": "reference", "reference_mode": mode, "reference_noise": 0.5}},
    })


# -- batching must not change results -------------------------------------------------


@pytest.mark.parametrize("protocol,condition,view", [
    ("full_direct", "FULL", "full_context"),
    ("oracle_transition", "SPLIT_STATEFUL", "split_stateful"),
    ("generated_rollout", "SPLIT_STATEFUL", "split_stateful"),
    ("oracle_trajectory", "SPLIT_STATEFUL", "split_stateful"),
])
def test_batch_size_does_not_change_records(dataset, dataset_root, tmp_path, protocol, condition, view):
    """A batched run and an unbatched run must produce identical scored records."""
    def records_for(batch_size):
        cfg = make_cfg(dataset_root, tmp_path, name=f"{protocol}_{batch_size}", condition=condition,
                       view=view, protocols=[protocol], batch_size=batch_size)
        eval_set = build_eval_set(dataset, cfg)
        backend = ReferenceBackend(cfg.evaluation.backend)
        return [
            (r.record_id, r.prediction, r.strict_correct, r.lenient_correct, r.failure_type, r.extraction_route)
            for r in run_protocol(backend, eval_set, cfg, protocol)
        ]

    assert records_for(1) == records_for(8)


def test_rollout_batching_keeps_states_separate(dataset, dataset_root, tmp_path):
    """Batched rollouts must never let one problem's state leak into another's prompt."""
    cfg = make_cfg(dataset_root, tmp_path, name="leak", condition="SPLIT_STATEFUL",
                   view="split_stateful", protocols=["generated_rollout"], batch_size=8, mode="oracle")
    eval_set = build_eval_set(dataset, cfg)
    backend = ReferenceBackend(cfg.evaluation.backend)
    for record in run_protocol(backend, eval_set, cfg, "generated_rollout"):
        trajectory = eval_set.trajectories[record.problem_id]
        steps = record.meta["rollout"]["steps"]
        assert steps[0]["state_in"] == trajectory.full.initial_state
        for previous, following in zip(steps, steps[1:]):
            assert following["state_in"] == previous["state_out"]


# -- the eval subset must be seed-independent -----------------------------------------


def test_eval_subset_is_identical_across_training_seeds(dataset, dataset_root, tmp_path):
    def ids(seed):
        cfg = make_cfg(dataset_root, tmp_path, name=f"subset{seed}", condition="FULL",
                       view="full_context", protocols=["full_direct"], batch_size=4, seed=seed)
        return build_eval_set(dataset, cfg).order

    assert ids(1) == ids(2) == ids(999)


def test_subset_seed_still_controls_the_subset(dataset, dataset_root, tmp_path):
    def ids(subset_seed):
        cfg = make_cfg(dataset_root, tmp_path, name=f"ss{subset_seed}", condition="FULL",
                       view="full_context", protocols=["full_direct"], batch_size=4)
        cfg.evaluation.subset_seed = subset_seed
        return build_eval_set(dataset, cfg).order

    assert ids(1) != ids(2)


def test_two_arms_record_the_same_problem_hash(dataset, dataset_root, tmp_path):
    full = make_cfg(dataset_root, tmp_path, name="hash_full", condition="FULL",
                    view="full_context", protocols=["full_direct"], batch_size=4, seed=1)
    split = make_cfg(dataset_root, tmp_path, name="hash_split", condition="SPLIT_STATEFUL",
                     view="split_stateful", protocols=["generated_rollout"], batch_size=4, seed=7)
    left = evaluate(full, dataset)["eval_set"]["problem_ids_sha256"]
    right = evaluate(split, dataset)["eval_set"]["problem_ids_sha256"]
    assert left == right


# -- reasoning blocks -----------------------------------------------------------------


def test_reasoning_blocks_are_stripped_end_to_end(dataset, dataset_root, tmp_path):
    """A <think> block containing a wrong answer must not be scored."""
    from rineya_lab.evaluation.backends import GenerationResult
    from rineya_lab.evaluation.protocols import run_protocol as run

    cfg = make_cfg(dataset_root, tmp_path, name="think", condition="FULL",
                   view="full_context", protocols=["full_direct"], batch_size=4, mode="oracle")
    eval_set = build_eval_set(dataset, cfg)

    class ThinkingBackend(ReferenceBackend):
        def generate(self, messages, **overrides):
            gold = self._oracle.get("answer", "?")
            text = f"<think>My first guess is 999999, but let me redo it.</think>\nFinal answer: {gold}"
            return GenerationResult(text=text, completion_tokens=12)

    backend = ThinkingBackend(cfg.evaluation.backend)
    records = list(run(backend, eval_set, cfg, "full_direct"))
    assert records and all(r.strict_correct for r in records)
    assert all(not r.components["parse"]["ambiguous"] for r in records)
    assert all(any("reasoning block" in n for n in r.components["parse"]["notes"]) for r in records)


def test_reasoning_blocks_can_be_disabled(dataset, dataset_root, tmp_path):
    cfg = make_cfg(dataset_root, tmp_path, name="nothink", condition="FULL",
                   view="full_context", protocols=["full_direct"], batch_size=4)
    cfg.evaluation.strip_reasoning_blocks = False
    assert cfg.evaluation.strip_reasoning_blocks is False


# -- cross-seed aggregation -----------------------------------------------------------


def _fake_report(tmp_path: Path, name: str, difference_pp: float, seed: int) -> Path:
    payload = {
        "settings": {"seed": seed, "non_inferiority_margin_pp": 2.0, "margin_frozen": True},
        "candidate": {"name": name},
        "comparisons": {
            "full_direct__vs__generated_rollout": {
                "n_paired": 200,
                "decision": {"superiority": "not_established"},
                "bootstrap": {
                    "observed_difference_pp": difference_pp,
                    "ci_low_pp": difference_pp - 3, "ci_high_pp": difference_pp + 3,
                    "baseline_mean": 0.80, "candidate_mean": 0.80 + difference_pp / 100,
                },
            }
        },
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_aggregate_combines_seeds(tmp_path):
    paths = [_fake_report(tmp_path, f"seed{i}", d, i) for i, d in enumerate([-0.5, 0.2, -0.1], start=1)]
    cfg = ComparisonConfig(margin_frozen=True)
    payload = aggregate_reports(paths, cfg, output_dir=tmp_path / "agg")
    summary = payload["comparisons"]["full_direct__vs__generated_rollout"]
    assert summary["seeds"] == 3
    assert summary["verdict"] == "non_inferior"
    assert (tmp_path / "agg" / "multiseed_report.md").exists()


def test_aggregate_flags_high_between_seed_variance(tmp_path):
    paths = [_fake_report(tmp_path, f"v{i}", d, i) for i, d in enumerate([-12.0, 9.0, 1.0], start=1)]
    summary = aggregate(
        collect_seed_entries(paths)["full_direct__vs__generated_rollout"],
        ComparisonConfig(margin_frozen=True),
    )
    assert summary["verdict"] == "inconclusive_at_this_power"
    assert any("between-seed SD" in line for line in summary["rationale"])
    assert any("disagree in sign" in line for line in summary["rationale"])


def test_aggregate_refuses_to_conclude_from_one_seed(tmp_path):
    paths = [_fake_report(tmp_path, "solo", -0.4, 1)]
    summary = aggregate(
        collect_seed_entries(paths)["full_direct__vs__generated_rollout"],
        ComparisonConfig(margin_frozen=True),
    )
    assert summary["verdict"] == "insufficient_seeds"
    assert summary["seed_level_ci_pp"] is None


def test_aggregate_warns_when_margin_not_frozen(tmp_path):
    paths = [_fake_report(tmp_path, f"w{i}", -0.2, i) for i in (1, 2, 3)]
    summary = aggregate(collect_seed_entries(paths)["full_direct__vs__generated_rollout"],
                        ComparisonConfig(margin_frozen=False))
    assert any("margin_frozen is false" in line for line in summary["rationale"])


# -- gate G9: learnability floor -------------------------------------------------------


def _report(path: Path, accuracy: float) -> Path:
    path.write_text(json.dumps({"protocols": {"full_direct": {"strict": {"accuracy": accuracy}}}}), encoding="utf-8")
    return path


def _load_gate():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "check_learnability.py"
    spec = importlib.util.spec_from_file_location("check_learnability", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_learnability_gate_blocks_a_floor_effect(tmp_path, monkeypatch):
    """A FULL arm stuck at the floor must stop the experiment, not proceed."""
    gate = _load_gate()
    monkeypatch.setenv("LEARNABILITY_FLOOR_PCT", "25")
    monkeypatch.setenv("LEARNABILITY_MIN_LIFT_PP", "15")
    full = _report(tmp_path / "full.json", 0.03)
    base = _report(tmp_path / "base.json", 0.00)
    assert gate.main(["x", str(full), str(base)]) == 1


def test_learnability_gate_blocks_insufficient_lift(tmp_path, monkeypatch):
    gate = _load_gate()
    monkeypatch.setenv("LEARNABILITY_FLOOR_PCT", "25")
    monkeypatch.setenv("LEARNABILITY_MIN_LIFT_PP", "15")
    # clears the absolute floor, but the base was already almost that good
    full = _report(tmp_path / "full.json", 0.30)
    base = _report(tmp_path / "base.json", 0.22)
    assert gate.main(["x", str(full), str(base)]) == 1


def test_learnability_gate_passes_a_healthy_run(tmp_path, monkeypatch):
    gate = _load_gate()
    monkeypatch.setenv("LEARNABILITY_FLOOR_PCT", "25")
    monkeypatch.setenv("LEARNABILITY_MIN_LIFT_PP", "15")
    full = _report(tmp_path / "full.json", 0.62)
    base = _report(tmp_path / "base.json", 0.01)
    assert gate.main(["x", str(full), str(base)]) == 0
