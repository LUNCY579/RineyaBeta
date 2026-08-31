from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.cross_architecture_graft.src.evidence import sha256_file
from experiments.cross_architecture_graft.src.inspect_data import inspect_jsonl
from experiments.cross_architecture_graft.src.materialize_campaign import materialize_campaign
from experiments.cross_architecture_graft.src.verify_plans import verify_plans


class DataToolTests(unittest.TestCase):
    def test_preregistered_plan_bundle_is_matched(self) -> None:
        result = verify_plans()
        self.assertEqual(result["status"], "PREFLIGHT_BUNDLE_PASS")
        self.assertLess(
            result["arms"]["nope_gqa"]["trainable_parameter_gap_vs_primary"],
            0.01,
        )
        self.assertLess(
            result["arms"]["swiglu"]["trainable_parameter_gap_vs_primary"],
            0.01,
        )

    def test_inspector_requires_complete_independent_group_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "open_train.jsonl"
            rows = [
                {
                    "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
                    "category": "c",
                    "template_id": f"template-{index}",
                }
                for index in range(12)
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            result = inspect_jsonl(path)
            self.assertEqual(result["status"], "READY_FOR_GROUPED_SPLIT")
            self.assertEqual(result["recommended_group_field_precedence"], ["template_id"])

    def test_campaign_materializer_produces_three_arms_for_each_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            train = root / "train.jsonl"
            dev = root / "dev.jsonl"
            split = root / "split.json"
            train.write_text('{"messages": [], "group_id": "a"}\n', encoding="utf-8")
            dev.write_text('{"messages": [], "group_id": "b"}\n', encoding="utf-8")
            split.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "group_overlap": 0,
                        "train": {"rows": 1, "groups": 1, "sha256": sha256_file(train)},
                        "dev": {"rows": 1, "groups": 1, "sha256": sha256_file(dev)},
                    }
                ),
                encoding="utf-8",
            )
            output = root / "campaign"
            result = materialize_campaign(
                model_path=model,
                train_mix=train,
                dev_mix=dev,
                split_manifest=split,
                output_dir=output,
                seeds=(11, 13),
                max_train_rows=1,
                max_dev_rows=1,
            )
            self.assertEqual(result["status"], "CONFIGS_MATERIALIZED_UNRUN")
            self.assertEqual(len(result["generated_configs"]), 6)
            self.assertFalse(result["sealed_evaluation_opened"])
            for record in result["generated_configs"]:
                self.assertTrue(Path(record["path"]).is_file())


if __name__ == "__main__":
    unittest.main()
