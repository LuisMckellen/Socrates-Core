"""Offline tests for run_session with scripted terminal input.

Run from the project root:  python -m unittest discover -s tests -v
No model, no NPU, no network.

run_session.py always loads the live question_bank.json (it has no bank
injection point), so these tests target real ids from the current
16-entry bank rather than a synthetic fixture.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run_session  # noqa: E402
from socratic_core.mock_client import MockInferenceClient  # noqa: E402

# s_cell_theory_L1: correct_answer/accepted_variants, plus its own two
# misconceptions (guaranteed on-topic and, being neither "division" nor
# "pre-existing", a clean key_terms Case C wrong answer).
QID_L1 = "s_cell_theory_L1"
CORRECT_L1 = (
    "Growth and repair occur because pre-existing functional cells undergo "
    "division to generate new cells."
)
THREE_WRONG_L1 = [
    "Wounds heal because damaged tissue cells absorb surrounding nutrients "
    "and swell in size until the empty space in the tissue is filled.",
    "New cells form spontaneously from an acellular matrix or fluid at the "
    "wound site to rebuild the missing tissue.",
    "The body simply grows more tissue whenever it needs to heal a wound.",
]

# s_cell_theory_L2: same idea, second question so a multi-question session
# can solve one and get stuck on the other.
QID_L2 = "s_cell_theory_L2"
THREE_WRONG_L2 = [
    "Mitochondria and chloroplasts are independent living organisms inside "
    "cells because they have their own DNA, ribosomes, and double membranes.",
    "Mitochondria are considered alive because they perform a life process "
    "and anything that performs a life process must itself be alive.",
    "The cell simply produces more parts whenever it grows bigger, without "
    "any special mechanism keeping it alive on its own.",
]


class RunSessionTests(unittest.TestCase):
    def setUp(self):
        self.sessions_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.dict(os.environ, {"SOCRATIC_SESSIONS_DIR": str(self.sessions_dir)}))
        self.enterContext(redirect_stdout(io.StringIO()))

    def test_session_runs_to_completion_with_mock(self):
        summary = run_session.main(
            ["--question-id", QID_L1],
            input_fn=lambda _, it=iter([THREE_WRONG_L1[0], CORRECT_L1]): next(it),
        )
        self.assertEqual(summary["solved_question_ids"], [QID_L1])
        self.assertEqual(summary["turns"], 2)

        logs = list(self.sessions_dir.glob("*.json"))
        self.assertEqual(len(logs), 1)
        log = json.loads(logs[0].read_text(encoding="utf-8"))
        self.assertIsNotNone(log["finished_at"])
        self.assertEqual(log["solved_question_ids"], [QID_L1])
        self.assertEqual(log["history"][-1]["event"], "session_finished")

    def test_session_logs_stuck_question_after_three_wrong_attempts(self):
        summary = run_session.main(
            ["--question-id", QID_L1], input_fn=lambda _, it=iter(THREE_WRONG_L1): next(it)
        )
        self.assertEqual(summary["stuck_question_ids"], [QID_L1])
        self.assertEqual(summary["solved_question_ids"], [])
        self.assertEqual(summary["turns"], 3)

        log = json.loads(next(self.sessions_dir.glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(log["stuck_question_ids"], [QID_L1])
        self.assertIn("escalation", [h["event"] for h in log["history"]])
        self.assertIsNotNone(log["finished_at"])

    def test_summary_reports_solved_and_stuck(self):
        summary = run_session.main(
            ["--question-id", QID_L1, QID_L2],
            input_fn=lambda _, it=iter([CORRECT_L1, *THREE_WRONG_L2]): next(it),
        )
        self.assertEqual(len(summary["solved_question_ids"]), 1)
        self.assertEqual(len(summary["stuck_question_ids"]), 1)
        self.assertEqual(summary["solved_question_ids"], [QID_L1])
        self.assertEqual(summary["stuck_question_ids"], [QID_L2])
        self.assertEqual(summary["turns"], 4)
        self.assertGreaterEqual(summary["elapsed_s"], 0.0)

    def test_default_client_is_mock(self):
        # No --question-id: run_session defaults to the bank's first id,
        # which is the f_001 filter ("Who named ... cells ... ?").
        with (
            patch.object(run_session, "MockInferenceClient", side_effect=MockInferenceClient) as mock_cls,
            patch.object(run_session, "InferenceClient") as npu_cls,
        ):
            summary = run_session.main([], input_fn=lambda _, it=iter(["Robert Hooke"]): next(it))
        mock_cls.assert_called_once()
        npu_cls.assert_not_called()
        self.assertEqual(summary["solved_question_ids"], ["f_001"])


if __name__ == "__main__":
    unittest.main()
