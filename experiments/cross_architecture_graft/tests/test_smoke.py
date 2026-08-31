from __future__ import annotations

import unittest
from pathlib import Path

from experiments.cross_architecture_graft.src.config import load_config
from experiments.cross_architecture_graft.src.smoke import run_smoke


ROOT = Path(__file__).resolve().parents[1]


class SmokeTests(unittest.TestCase):
    def test_end_to_end_smoke_passes_without_quality_claim(self) -> None:
        outcome = run_smoke(load_config(ROOT / "configs" / "smoke.json"))
        self.assertEqual(outcome.status, "INFRASTRUCTURE_PASS_ONLY")
        self.assertTrue(outcome.invariants["zero_gate_bit_exact"])
        self.assertTrue(outcome.invariants["host_parameters_unchanged"])
        self.assertTrue(outcome.invariants["frozen_donor_parameters_unchanged"])
        self.assertLess(
            outcome.results["candidate_final_dev_mse"],
            outcome.results["candidate_initial_dev_mse"],
        )


if __name__ == "__main__":
    unittest.main()
