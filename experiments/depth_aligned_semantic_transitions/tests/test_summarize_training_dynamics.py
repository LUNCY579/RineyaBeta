"""Offline tests for V2 training-dynamics summary and unsigned V2 configs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from dast.config import assert_split_allowed, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SCRIPT = ROOT / "scripts" / "summarize_training_dynamics.py"


def _load_summarizer():
    spec = importlib.util.spec_from_file_location("summarize_training_dynamics", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trapezoid_auc_matches_hand_calculation():
    mod = _load_summarizer()
    assert mod.trapezoid_auc([0.0, 2.0, 4.0], [1.0, 3.0, 3.0]) == pytest.approx(10.0)
    assert mod.trapezoid_auc([1.0], [1.0]) is None


def _write_run(root: Path, name: str, arm: str, seed: int, *, l_state: float, lm_path: list[tuple[int, float]]) -> None:
    directory = root / name
    directory.mkdir()
    (directory / "run_manifest.json").write_text(
        json.dumps({"arm": arm, "seed": seed}),
        encoding="utf-8",
    )
    metrics = [
        {"optimizer_step": step, "loss/lm": lm, "loss/state": l_state + 0.01}
        for step, lm in lm_path
    ]
    (directory / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metrics),
        encoding="utf-8",
    )
    (directory / "dev_summary.json").write_text(
        json.dumps({"loss/state": l_state, "loss/lm": lm_path[-1][1], "lm/perplexity": 1.01}),
        encoding="utf-8",
    )


def test_v2_gates_pass_on_constructed_conflict(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    mod = _load_summarizer()
    runs = tmp_path / "runs"
    runs.mkdir()
    # Ordered harder to fit, higher LM-AUC; reverse and final-repeated easier.
    for seed, prefix in ((17, "v2_s17"), (29, "v2_s29"), (43, "v2_s43")):
        _write_run(
            runs,
            f"{prefix}_state_ordered",
            "state_ordered",
            seed,
            l_state=0.260,
            lm_path=[(0, 2.0), (250, 1.2)],
        )
        _write_run(
            runs,
            f"{prefix}_state_reverse",
            "state_reverse",
            seed,
            l_state=0.250,
            lm_path=[(0, 2.0), (250, 1.0)],
        )
        _write_run(
            runs,
            f"{prefix}_state_final_repeated",
            "state_final_repeated",
            seed,
            l_state=0.140,
            lm_path=[(0, 2.0), (250, 1.0)],
        )

    code = mod.main(["--prefix", "v2_"], runs_dir=runs)
    assert code == 0
    found, pending = mod.collect_runs(runs, ["v2_"])
    assert pending == []
    gates = mod.evaluate_v2_gates(found)
    assert gates["C1"]["mean"] == pytest.approx(0.010)
    assert gates["C2"]["mean"] == pytest.approx(0.120)
    assert gates["V2_G1"] is True
    assert gates["V2_G2"] is True
    out = capsys.readouterr().out
    assert "V2-G1 (C1 and C2): PASS" in out
    assert "CLAIM BOUNDARY" in out


def test_v2_g1_fails_when_ordered_is_easier(tmp_path: Path):
    mod = _load_summarizer()
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run(runs, "v2_s17_state_ordered", "state_ordered", 17, l_state=0.20, lm_path=[(0, 1.0), (10, 1.0)])
    _write_run(runs, "v2_s17_state_reverse", "state_reverse", 17, l_state=0.26, lm_path=[(0, 1.0), (10, 1.0)])
    _write_run(
        runs,
        "v2_s17_state_final_repeated",
        "state_final_repeated",
        17,
        l_state=0.14,
        lm_path=[(0, 1.0), (10, 1.0)],
    )
    found, _pending = mod.collect_runs(runs, ["v2_"])
    gates = mod.evaluate_v2_gates(found)
    assert gates["V2_G1"] is False
    assert gates["C1"]["pass"] is False


@pytest.mark.parametrize(
    "name,arm,seed",
    [
        ("v2_s17_state_ordered.yaml", "state_ordered", 17),
        ("v2_s29_state_reverse.yaml", "state_reverse", 29),
        ("v2_s43_state_final_repeated.yaml", "state_final_repeated", 43),
    ],
)
def test_v2_configs_load_unsigned_and_keep_test_sealed(name: str, arm: str, seed: int):
    cfg = load_config(CONFIGS / name)
    assert cfg.experiment.arm == arm
    assert cfg.experiment.seed == seed
    assert cfg.experiment.name.startswith("dast-v2-")
    assert cfg.experiment.preregistration_frozen is False
    assert cfg.experiment.preregistration_sha256 == ""
    assert cfg.experiment.allow_sealed_test is False
    with pytest.raises(PermissionError, match="TEST is sealed"):
        assert_split_allowed(cfg, "test")
