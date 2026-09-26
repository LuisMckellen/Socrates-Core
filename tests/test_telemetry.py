"""Tests for socratic_core.telemetry: pure counters over answer events.

Run from the project root:  python -m pytest tests/test_telemetry.py
No model, no NPU, no network.
"""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from socratic_core.telemetry import bank_usage, case_a_failure_rate, cloud_fallback_rate  # noqa: E402


def _answer(**fields):
    return {"event": "answer", **fields}


class TelemetryTests(unittest.TestCase):
    def test_bank_usage_counts(self):
        turns = [
            _answer(matched_bank_id="ct_hypertrophy"),
            _answer(matched_bank_id="ct_hypertrophy"),
            _answer(matched_bank_id="ct_spontaneous_generation"),
            _answer(matched_bank_id="natural_correct"),
            _answer(matched_bank_id="partial_example"),
            # Non-answer events never count, whatever they carry.
            {"event": "llm_classify", "matched_bank_id": "ct_hypertrophy"},
        ]
        usage = bank_usage(turns)
        self.assertEqual(usage, Counter({"ct_hypertrophy": 2, "ct_spontaneous_generation": 1, "natural_correct": 1, "partial_example": 1}))
        self.assertTrue(all(isinstance(k, str) for k in usage))

    def test_bank_usage_skips_none(self):
        turns = [_answer(matched_bank_id=None), _answer(matched_bank_id="none"), _answer(matched_bank_id="ct_spontaneous_generation")]
        self.assertEqual(bank_usage(turns), Counter({"ct_spontaneous_generation": 1}))

    def test_telemetry_empty_input(self):
        self.assertEqual(bank_usage([]), Counter())
        self.assertEqual(case_a_failure_rate([]), 0.0)
        self.assertEqual(cloud_fallback_rate([]), 0.0)
        # History with no answer events behaves the same.
        no_answers = [{"event": "question_presented"}, {"event": "session_finished"}]
        self.assertEqual(bank_usage(no_answers), Counter())
        self.assertEqual(case_a_failure_rate(no_answers), 0.0)
        self.assertEqual(cloud_fallback_rate(no_answers), 0.0)

    def test_telemetry_handles_missing_field(self):
        # Logs persisted before the fields existed: no matched_bank_id,
        # case_a_verified or cloud_fallback_used anywhere.
        legacy = [_answer(verdict="wrong"), _answer(verdict="correct"), {"ts": "x"}]
        self.assertEqual(bank_usage(legacy), Counter())
        self.assertEqual(case_a_failure_rate(legacy), 0.0)
        self.assertEqual(cloud_fallback_rate(legacy), 0.0)
        # Mixed with new-shape entries: missing fields count as None.
        mixed = legacy + [_answer(matched_bank_id="ct_hypertrophy", case_a_verified=False, cloud_fallback_used=True)]
        self.assertEqual(bank_usage(mixed), Counter({"ct_hypertrophy": 1}))
        self.assertEqual(case_a_failure_rate(mixed), 1.0)
        self.assertAlmostEqual(cloud_fallback_rate(mixed), 1 / 3)

    def test_case_a_failure_rate_zero_when_no_case_a(self):
        turns = [_answer(case_a_verified=None), _answer(case_a_verified=None), _answer()]
        self.assertEqual(case_a_failure_rate(turns), 0.0)

    def test_case_a_failure_rate_mixed(self):
        turns = [
            _answer(case_a_verified=True),
            _answer(case_a_verified=False),
            _answer(case_a_verified=True),
            _answer(case_a_verified=False),
            _answer(case_a_verified=False),
            _answer(case_a_verified=None),  # not a Case A turn: excluded from the denominator
        ]
        self.assertAlmostEqual(case_a_failure_rate(turns), 3 / 5)

    def test_cloud_fallback_rate_mixed(self):
        turns = [
            _answer(cloud_fallback_used=True),
            _answer(cloud_fallback_used=False),
            _answer(cloud_fallback_used=False),
            _answer(),  # missing -> not a fallback, still an answer
            {"event": "llm_classify", "cloud_fallback_used": True},  # not an answer event
        ]
        self.assertAlmostEqual(cloud_fallback_rate(turns), 1 / 4)


if __name__ == "__main__":
    unittest.main()
