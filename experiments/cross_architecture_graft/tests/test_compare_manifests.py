from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from experiments.cross_architecture_graft.src.compare_manifests import compare_manifests
from experiments.cross_architecture_graft.src.config import load_config
from experiments.cross_architecture_graft.src.evidence import atomic_write_json, build_manifest


ROOT = Path(__file__).resolve().parents[1]


def _write_config(root: Path, name: str, raw: dict) -> Path:
    path = root / f"{name}.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def _write_manifest(root: Path, name: str, config_path: Path, *, after: float) -> Path:
    config = load_config(config_path)
    manifest = build_manifest(
        config=config,
        status="UNSEALED_MECHANISM_RUN_COMPLETE",
        phase="real_train_dev",
        results={
            "dev_before": {"mean_nll": 1.0},
            "dev_after": {"mean_nll": after},
            "train_seconds": 1.0,
            "peak_allocated_mib": None,
        },
        arms=[
            {
                "name": config.experiment_id,
                "graft_parameters": 100,
                "graft_trainable_parameters": 100,
            }
        ],
        invariants={
            "zero_gate_bit_exact": True,
            "only_graft_parameters_trainable": True,
            "frozen_parameters_unchanged": True,
            "sealed_evaluation_opened": False,
        },
        data_assets=[
            {"role": "train", "sha256": "a" * 64},
            {"role": "dev", "sha256": "b" * 64},
            {"role": "split_manifest", "sha256": "c" * 64},
        ],
        model_assets=[{"aggregate_sha256": "d" * 64}],
    )
    path = root / f"{name}-manifest.json"
    atomic_write_json(path, manifest)
    return path


class CompareManifestTests(unittest.TestCase):
    def _pair(self, root: Path) -> tuple[Path, Path]:
        base = json.loads((ROOT / "configs" / "real_local_template.json").read_text())
        candidate = deepcopy(base)
        candidate["experiment_id"] = "candidate"
        candidate["output"]["run_dir"] = "runs/candidate"

        control = deepcopy(base)
        control["experiment_id"] = "control"
        control["output"]["run_dir"] = "runs/control"
        control_plan = json.loads(
            (ROOT / "configs" / "swiglu_matched_control_plan.json").read_text()
        )
        control["graft"] = control_plan["graft"]
        candidate_path = _write_config(root, "candidate", candidate)
        control_path = _write_config(root, "control", control)
        return (
            _write_manifest(root, "candidate", candidate_path, after=0.8),
            _write_manifest(root, "control", control_path, after=0.9),
        )

    def test_comparable_arms_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate, control = self._pair(Path(tmp))
            result = compare_manifests(
                [candidate, control], reference_id="candidate"
            )
            self.assertEqual(result["status"], "COMPARABILITY_PASS")
            self.assertFalse(result["sealed_evaluation_opened"])
            self.assertEqual(len(result["arms"]), 2)

    def test_training_contract_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, control = self._pair(root)
            control_manifest = json.loads(control.read_text())
            snapshot = control_manifest["config"]["snapshot"]
            snapshot["training"]["steps"] += 1
            # Rebuild so the file is internally valid; the comparator must catch
            # the cross-arm contract difference rather than a payload corruption.
            control_config = _write_config(root, "drift", snapshot)
            drift = _write_manifest(root, "drift", control_config, after=0.9)
            with self.assertRaisesRegex(ValueError, "differ outside"):
                compare_manifests([candidate, drift])


if __name__ == "__main__":
    unittest.main()
