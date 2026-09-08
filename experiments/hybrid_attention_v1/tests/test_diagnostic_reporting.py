"""Reporting-only tests; no pretrained model, dataset, optimizer, or sealed asset."""
import unittest
from ..diagnostics import DIAGNOSTIC_SCHEMA_VERSION, score, summarize


class DiagnosticReportingTests(unittest.TestCase):
    def _case(self, *, idx=0, family="retrieval", tokens=511, depth=0.0,
              answer="42", answerable=True):
        return {"id": f"dev-{idx:04d}", "family": family, "depth": depth,
                "tokens": tokens, "answerable": answerable, "prompt_sha256": str(idx),
                "answer": answer, "evidence_span": [20, 30], "question_span": [500, 511],
                "evidence_to_question_gap": 470, "evidence_kind": "support_or_conflict"}

    def _generation(self, ids=(7, 7, 8), *, eos=False, exhausted=True):
        return {"token_ids": list(ids), "stopped_eos": eos, "budget_exhausted": exhausted}

    def test_score_carries_debug_metadata_without_changing_exactness(self):
        row = score(self._case(), "42", self._generation())
        self.assertEqual(row["diagnostic_schema_version"], DIAGNOSTIC_SCHEMA_VERSION)
        self.assertTrue(row["strict_exact"])
        self.assertTrue(row["trimmed_exact"])
        self.assertEqual(row["evidence_span"], [20, 30])
        self.assertEqual(row["question_span"], [500, 511])
        self.assertEqual(row["evidence_to_question_gap"], 470)
        self.assertEqual(row["unique_generated_tokens"], 2)
        self.assertEqual(row["max_token_repeat"], 2)
        self.assertEqual(row["termination"], "budget")

    def test_eos_termination_is_distinct_from_budget(self):
        row = score(self._case(), "42", self._generation(ids=(7,), eos=True, exhausted=False))
        self.assertEqual(row["termination"], "eos")
        self.assertTrue(row["stopped_eos"])
        self.assertFalse(row["budget_exhausted"])

    def test_summary_adds_non_singleton_debugging_margins(self):
        rows = []
        specs = [("retrieval", 511, 0.0, True, "42", "42"),
                 ("retrieval", 513, 1.0, True, "42", "UNKNOWN"),
                 ("absent", 511, 0.0, False, "UNKNOWN", "UNKNOWN"),
                 ("absent", 513, 1.0, False, "UNKNOWN", "x")]
        for i, (family, tokens, depth, answerable, answer, text) in enumerate(specs):
            case = self._case(idx=i, family=family, tokens=tokens, depth=depth,
                              answer=answer, answerable=answerable)
            rows.append(score(case, text, self._generation(ids=(i, i, 9))))
        report = summarize(rows)
        self.assertEqual(report["overall"]["n"], 4)
        self.assertEqual(report["margins"]["by_family"]["retrieval"]["n"], 2)
        self.assertEqual(report["margins"]["by_family"]["absent"]["n"], 2)
        self.assertEqual(report["margins"]["by_length"]["511"]["n"], 2)
        self.assertEqual(report["margins"]["by_depth"]["1.0"]["n"], 2)
        self.assertEqual(report["margins"]["by_answerability"]["answerable"]["n"], 2)
        self.assertEqual(report["overall"]["max_token_repeat_max"], 2)
        self.assertIn("diagnostic-only", report["interpretation"])

    def test_empty_summary_remains_defined(self):
        report = summarize([])
        self.assertEqual(report["overall"]["n"], 0)
        self.assertIsNone(report["overall"]["strict_exact"])
        self.assertEqual(report["margins"], {"by_family": {}, "by_length": {},
                                             "by_depth": {}, "by_answerability": {}})


if __name__ == "__main__":
    unittest.main()
