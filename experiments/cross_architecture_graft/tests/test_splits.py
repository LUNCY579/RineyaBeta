from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.cross_architecture_graft.src.splits import build_grouped_split


class SplitTests(unittest.TestCase):
    def test_grouped_split_is_deterministic_and_has_zero_overlap(self) -> None:
        rows = []
        for group in range(30):
            for item in range(3):
                rows.append(
                    {
                        "source_family": f"family-{group}",
                        "category": f"c{group % 3}",
                        "messages": [{"role": "user", "content": f"{group}-{item}"}],
                    }
                )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            first = build_grouped_split(
                input_path=source,
                train_path=root / "train.jsonl",
                dev_path=root / "dev.jsonl",
                manifest_path=root / "manifest.json",
                group_fields=["source_family"],
                dev_fraction=0.2,
                seed=17,
            )
            train_bytes = (root / "train.jsonl").read_bytes()
            dev_bytes = (root / "dev.jsonl").read_bytes()
            second = build_grouped_split(
                input_path=source,
                train_path=root / "train2.jsonl",
                dev_path=root / "dev2.jsonl",
                manifest_path=root / "manifest2.json",
                group_fields=["source_family"],
                dev_fraction=0.2,
                seed=17,
            )
            self.assertEqual(first["group_overlap"], 0)
            self.assertEqual(first["train"]["sha256"], second["train"]["sha256"])
            self.assertEqual(first["dev"]["sha256"], second["dev"]["sha256"])
            self.assertEqual(train_bytes, (root / "train2.jsonl").read_bytes())
            self.assertEqual(dev_bytes, (root / "dev2.jsonl").read_bytes())

    def test_missing_group_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            source.write_text(json.dumps({"category": "x"}) + "\n")
            with self.assertRaisesRegex(ValueError, "refusing a random row split"):
                build_grouped_split(
                    input_path=source,
                    train_path=root / "train.jsonl",
                    dev_path=root / "dev.jsonl",
                    manifest_path=root / "manifest.json",
                    group_fields=["source_family"],
                    dev_fraction=0.2,
                    seed=17,
                )


if __name__ == "__main__":
    unittest.main()
