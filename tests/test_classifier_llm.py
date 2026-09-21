"""Offline tests for classifier_llm against the mock client.

Run from the project root:  python -m unittest discover -s tests -v
No model, no NPU, no network.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from socratic_core.classifier_llm import classify_llm  # noqa: E402
from socratic_core.mock_client import MockInferenceClient  # noqa: E402
from socratic_core.question_bank import load_question_bank  # noqa: E402

BANK = load_question_bank(ROOT / "question_bank.json")
Q1 = BANK.get("s_cell_theory_L1")


class _GarbageClient(MockInferenceClient):
    def _canned_text(self, prompt: str) -> str:
        return "asdf qwerty 12345 ???"


class ClassifierLLMTests(unittest.TestCase):
    def test_clear_wording_error_returns_wording(self):
        mock = MockInferenceClient()
        r = classify_llm("this is a wording mistake", Q1, mock)
        self.assertEqual(r["error_type"], "wording_error")
        self.assertGreaterEqual(r["confidence"], 0.7)

    def test_clear_logic_error_returns_logic(self):
        mock = MockInferenceClient()
        r = classify_llm("this is a logic mistake", Q1, mock)
        self.assertEqual(r["error_type"], "logic_error")
        self.assertGreaterEqual(r["confidence"], 0.7)

    def test_low_confidence_falls_back_to_logic(self):
        mock = MockInferenceClient(force_confidence=0.5)
        r = classify_llm("this is a wording mistake", Q1, mock)
        self.assertEqual(r["error_type"], "logic_error")
        self.assertEqual(r["confidence"], 0.5)
        self.assertTrue(r["reasoning"])

    def test_client_error_falls_back_to_logic_zero_conf(self):
        mock = MockInferenceClient(fail_mode=True)
        r = classify_llm("the organ", Q1, mock)
        self.assertEqual(r["error_type"], "logic_error")
        self.assertEqual(r["confidence"], 0.0)
        self.assertEqual(r["reasoning"], "client error")

    def test_unparseable_output_falls_back_to_logic(self):
        mock = _GarbageClient()
        r = classify_llm("the organ", Q1, mock)
        self.assertEqual(r["error_type"], "logic_error")
        self.assertEqual(r["confidence"], 0.0)
        self.assertEqual(r["reasoning"], "unparseable output")

    def test_mock_call_log_records_invocation(self):
        mock = MockInferenceClient()
        classify_llm("the organ", Q1, mock)
        self.assertEqual(len(mock.call_log), 1)
        entry = mock.call_log[0]
        self.assertIn("the organ", entry["prompt"])
        self.assertEqual(set(entry["response"]), {"text", "ttft_ms", "total_ms", "tokens_generated", "error"})


if __name__ == "__main__":
    unittest.main()
