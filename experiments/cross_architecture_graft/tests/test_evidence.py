from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.cross_architecture_graft.src.config import load_config
from experiments.cross_architecture_graft.src.evidence import (
    atomic_write_json,
    build_manifest,
    verify_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


class EvidenceTests(unittest.TestCase):
    def test_manifest_hard_codes_claim_boundary_and_sealed_false(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.json")
        manifest = build_manifest(
            config=config,
            status="INFRASTRUCTURE_PASS_ONLY",
            phase="unit",
            results={},
            arms=[],
            invariants={"sealed_evaluation_opened": False},
        )
        self.assertFalse(manifest["sealed_evaluation_opened"])
        self.assertIn("does not establish", manifest["claim_boundary"])
        self.assertEqual(len(manifest["manifest_payload_sha256"]), 64)


    def test_manifest_snapshot_and_payload_are_verified(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.json")
        manifest = build_manifest(
            config=config,
            status="INFRASTRUCTURE_PASS_ONLY",
            phase="unit",
            results={},
            arms=[],
            invariants={"sealed_evaluation_opened": False},
        )
        verify_manifest(manifest)
        manifest["status"] = "QUALITY_PASS"
        with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
            verify_manifest(manifest)

    def test_manifest_cannot_claim_sealed_opened(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.json")
        manifest = build_manifest(
            config=config,
            status="INFRASTRUCTURE_PASS_ONLY",
            phase="unit",
            results={},
            arms=[],
            invariants={"sealed_evaluation_opened": False},
        )
        manifest["sealed_evaluation_opened"] = True
        with self.assertRaisesRegex(ValueError, "sealed evaluation"):
            verify_manifest(manifest)

    def test_atomic_json_write_leaves_no_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "manifest.json"
            atomic_write_json(path, {"b": 2, "a": 1})
            self.assertEqual(json.loads(path.read_text()), {"a": 1, "b": 2})
            self.assertEqual(list(path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
