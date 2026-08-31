from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.cross_architecture_graft.src.evidence import sha256_file
from experiments.cross_architecture_graft.src.real_runner import _verify_split_manifest


class RealRunnerGuardTests(unittest.TestCase):
    def test_split_manifest_hashes_and_zero_overlap_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.jsonl"
            dev = root / "dev.jsonl"
            split = root / "split.json"
            train.write_text('{"x": 1}\n', encoding="utf-8")
            dev.write_text('{"x": 2}\n', encoding="utf-8")
            split.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "group_overlap": 0,
                        "train": {
                            "rows": 1,
                            "groups": 1,
                            "sha256": sha256_file(train),
                        },
                        "dev": {
                            "rows": 1,
                            "groups": 1,
                            "sha256": sha256_file(dev),
                        },
                    }
                ),
                encoding="utf-8",
            )
            verified = _verify_split_manifest(split, train, dev)
            self.assertEqual(verified["group_overlap"], 0)
            dev.write_text('{"x": 3}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from the frozen split"):
                _verify_split_manifest(split, train, dev)

    def test_nonzero_group_overlap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.jsonl"
            dev = root / "dev.jsonl"
            split = root / "split.json"
            train.write_text("{}\n", encoding="utf-8")
            dev.write_text("{}\n", encoding="utf-8")
            split.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "group_overlap": 1,
                        "train": {"rows": 1, "groups": 1, "sha256": sha256_file(train)},
                        "dev": {"rows": 1, "groups": 1, "sha256": sha256_file(dev)},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "zero group overlap"):
                _verify_split_manifest(split, train, dev)


if __name__ == "__main__":
    unittest.main()
