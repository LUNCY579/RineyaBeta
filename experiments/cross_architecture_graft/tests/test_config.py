from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.cross_architecture_graft.src.config import load_config, validate_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_smoke_config_loads_and_hash_is_stable(self) -> None:
        first = load_config(ROOT / "configs" / "smoke.json")
        second = load_config(ROOT / "configs" / "smoke.json")
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first.mode, "smoke")
        self.assertFalse(first.raw["data_policy"]["allow_sealed"])

    def test_unknown_field_is_rejected(self) -> None:
        raw = json.loads((ROOT / "configs" / "smoke.json").read_text())
        raw["mystery"] = 1
        with self.assertRaisesRegex(ValueError, "unknown top-level"):
            validate_config(raw)

    def test_explicit_unsealed_paths_are_allowed(self) -> None:
        raw = json.loads((ROOT / "configs" / "real_local_template.json").read_text())
        validated = validate_config(raw)
        self.assertIn("unsealed_train", validated["data"]["train_mix"])

    def test_sealed_path_is_rejected_even_when_nested(self) -> None:
        raw = json.loads((ROOT / "configs" / "real_local_template.json").read_text())
        raw["data"]["dev_mix"] = "artifacts/eval/sealed/final.jsonl"
        with self.assertRaisesRegex(ValueError, "sealed/hidden"):
            validate_config(raw)

    def test_claim_level_cannot_be_promoted_by_config(self) -> None:
        raw = json.loads((ROOT / "configs" / "smoke.json").read_text())
        raw["claim_level"] = "quality_established"
        with self.assertRaisesRegex(ValueError, "infrastructure_only"):
            validate_config(raw)


    def test_donor_width_must_match_explicit_foreign_width(self) -> None:
        raw = json.loads((ROOT / "configs" / "smoke.json").read_text())
        raw["graft"]["donor"]["d_model"] = 8
        with self.assertRaisesRegex(ValueError, "must equal graft.foreign_dim"):
            validate_config(raw)

    def test_v1_cannot_unfreeze_host_or_disable_exact_bypass(self) -> None:
        raw = json.loads((ROOT / "configs" / "smoke.json").read_text())
        raw["graft"]["freeze_host"] = False
        with self.assertRaisesRegex(ValueError, "freeze_host=true"):
            validate_config(raw)
        raw["graft"]["freeze_host"] = True
        raw["graft"]["exact_eval_bypass"] = False
        with self.assertRaisesRegex(ValueError, "exact_eval_bypass=true"):
            validate_config(raw)

    def test_real_mode_requires_frozen_split_manifest(self) -> None:
        raw = json.loads((ROOT / "configs" / "real_local_template.json").read_text())
        del raw["data"]["split_manifest"]
        with self.assertRaisesRegex(ValueError, "split_manifest"):
            validate_config(raw)

    def test_content_hashing_is_mandatory(self) -> None:
        raw = json.loads((ROOT / "configs" / "smoke.json").read_text())
        raw["data_policy"]["require_content_hashes"] = False
        with self.assertRaisesRegex(ValueError, "require_content_hashes=true"):
            validate_config(raw)

    def test_whitespace_does_not_change_semantic_hash(self) -> None:
        source = ROOT / "configs" / "smoke.json"
        raw = json.loads(source.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            alternate = Path(tmp) / "alternate.json"
            alternate.write_text(json.dumps(raw, separators=(",", ":")))
            self.assertEqual(load_config(source).sha256, load_config(alternate).sha256)


if __name__ == "__main__":
    unittest.main()
