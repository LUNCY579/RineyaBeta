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


class ReadoutV2Tests(unittest.TestCase):
    """The first P3 run scored 0/72 on every arm because the model answered and
    then overran. These pin the amendment that makes the endpoint resolvable."""

    def _case(self, answer="42"):
        return {"id": "dev-0000", "family": "retrieval", "depth": 0.0, "tokens": 512,
                "answerable": answer != "UNKNOWN", "prompt_sha256": "x", "answer": answer,
                "evidence_span": [20, 30], "question_span": [500, 512],
                "evidence_to_question_gap": 470, "evidence_kind": "support_or_conflict"}

    def _gen(self, *, eos=False, exhausted=True, stop=None, kind=None, ids=(7, 7, 8)):
        g = {"token_ids": list(ids), "stopped_eos": eos, "budget_exhausted": exhausted,
             "stop_token_id": stop}
        if kind is not None:
            g["stop_kind"] = kind
        return g

    def test_the_exact_failure_that_floored_the_first_run(self):
        # A0_global on dev-0000: correct answer, then overrun to the cap.
        row = score(self._case("64285"), " 64285\nAnswer: 64285\nAnswer: 64285", self._gen())
        self.assertFalse(row["trimmed_exact"])       # v1 endpoint: blind
        self.assertFalse(row["strict_exact"])
        self.assertTrue(row["first_content_exact"])  # v2 endpoint: resolves it
        self.assertEqual(row["first_line"], "64285")
        self.assertEqual(row["termination"], "budget")
        self.assertFalse(row["terminated"])

    def test_a_cleanly_terminated_answer_satisfies_both_endpoints(self):
        row = score(self._case("64285"), " 64285",
                    self._gen(eos=True, exhausted=False, stop=198, kind="stop_sequence"))
        self.assertTrue(row["trimmed_exact"])
        self.assertTrue(row["first_content_exact"])
        self.assertTrue(row["terminated"])
        self.assertEqual(row["termination"], "stop_sequence")
        self.assertEqual(row["stop_token_id"], 198)

    def test_stop_kinds_stay_distinguishable(self):
        for kind, stop in (("eos", 32744), ("stop_sequence", 198)):
            with self.subTest(kind=kind):
                row = score(self._case(), "42",
                            self._gen(eos=True, exhausted=False, stop=stop, kind=kind))
                self.assertEqual(row["termination"], kind)

    def test_a_wrong_first_line_is_not_rescued_by_later_text(self):
        # The endpoint must not become "the answer appears somewhere".
        row = score(self._case("42"), "I think\n42", self._gen())
        self.assertFalse(row["first_content_exact"])
        self.assertFalse(row["trimmed_exact"])

    def test_abstention_is_read_from_the_first_line_too(self):
        row = score(self._case("UNKNOWN"), " UNKNOWN\nAnswer: UNKNOWN", self._gen())
        self.assertTrue(row["abstained"])
        self.assertTrue(row["first_content_exact"])
        self.assertFalse(row["trimmed_exact"])

    def test_cell_reports_both_endpoints_and_termination(self):
        rows = [score(self._case("42"), " 42\nAnswer: 42", self._gen()),
                score(self._case("42"), " 42",
                      self._gen(eos=True, exhausted=False, stop=198, kind="stop_sequence"))]
        cell = summarize(rows)["overall"]
        self.assertEqual(cell["first_content_exact"], 1.0)
        self.assertEqual(cell["trimmed_exact"], 0.5)
        self.assertEqual(cell["terminated_rate"], 0.5)
        self.assertEqual(cell["answerable_accuracy"], 1.0)
        self.assertEqual(cell["answerable_accuracy_trimmed"], 0.5)


class StopTokenContentTests(unittest.TestCase):
    """A newline stop token can carry CONTENT. Dropping the whole token turns
    'UNKNOWN.' into 'UNKNOWN' and scores a wrong answer correct -- and it does
    so at different rates per arm, so it would bias the contrasts. Caught by
    comparing readout v1 and v2 first lines: A0 matched 72/72 but A1 only
    26/72, every mismatch of this exact shape."""

    def _case(self, answer="UNKNOWN"):
        return {"id": "dev-0018", "family": "absent", "depth": 0.0, "tokens": 512,
                "answerable": False, "prompt_sha256": "x", "answer": answer,
                "evidence_span": [20, 30], "question_span": [500, 512],
                "evidence_to_question_gap": 470, "evidence_kind": "distractor"}

    def _readout(self, token_texts, stop_kind):
        """Reproduce run.py's trimming rule over decoded token pieces."""
        pieces = list(token_texts)
        if stop_kind == "eos":
            pieces = pieces[:-1]
        return "".join(pieces)

    def test_content_bearing_stop_token_is_kept(self):
        # Model emitted 'UNKNOWN' then the single token '.\n'.
        text = self._readout([" UNKNOWN", ".\n"], "stop_sequence")
        row = score(self._case(), text,
                    {"token_ids": [1, 2], "stopped_eos": True, "budget_exhausted": False,
                     "stop_token_id": 280, "stop_kind": "stop_sequence"})
        self.assertEqual(row["first_line"], "UNKNOWN.")
        self.assertFalse(row["first_content_exact"])  # 'UNKNOWN.' != 'UNKNOWN'
        self.assertFalse(row["trimmed_exact"])
        self.assertFalse(row["abstained"])

    def test_pure_special_stop_token_is_dropped(self):
        text = self._readout([" UNKNOWN", "<|im_end|>"], "eos")
        row = score(self._case(), text,
                    {"token_ids": [1, 2], "stopped_eos": True, "budget_exhausted": False,
                     "stop_token_id": 32744, "stop_kind": "eos"})
        self.assertEqual(row["first_line"], "UNKNOWN")
        self.assertTrue(row["first_content_exact"])
        self.assertTrue(row["abstained"])

    def test_a_bare_newline_stop_leaves_the_answer_intact(self):
        text = self._readout([" 64285", "\n"], "stop_sequence")
        row = score(self._case("64285"), text,
                    {"token_ids": [1, 2], "stopped_eos": True, "budget_exhausted": False,
                     "stop_token_id": 198, "stop_kind": "stop_sequence"})
        self.assertEqual(row["first_line"], "64285")
        self.assertTrue(row["first_content_exact"])
        self.assertTrue(row["trimmed_exact"])
