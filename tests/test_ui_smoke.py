"""Smoke tests for the Streamlit UI layer.

Covers the parts of ``ui/`` that are plain Python: module import, cluster
filtering and one full turn against the mock backend. Streamlit rendering is
deliberately not exercised -- it needs a script run context, and the widget
code here holds no logic worth asserting on.

Run from the project root:  python -m pytest tests/test_ui_smoke.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from socratic_core.mock_client import MockInferenceClient  # noqa: E402
from socratic_core.question_bank import load_question_bank  # noqa: E402
from socratic_core.state_machine import ENV_SESSIONS_DIR  # noqa: E402
from ui import state  # noqa: E402


class TestUISmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bank = load_question_bank()

    def setUp(self) -> None:
        # Keep session autosave out of the real sessions/ directory.
        self._tmp = tempfile.TemporaryDirectory()
        self._prev = os.environ.get(ENV_SESSIONS_DIR)
        os.environ[ENV_SESSIONS_DIR] = self._tmp.name

    def tearDown(self) -> None:
        if self._prev is None:
            os.environ.pop(ENV_SESSIONS_DIR, None)
        else:
            os.environ[ENV_SESSIONS_DIR] = self._prev
        self._tmp.cleanup()

    def test_app_module_imports(self) -> None:
        """Importing app must be side-effect free (rendering lives in main)."""
        import app

        self.assertTrue(callable(app.main))
        for label in state.BACKEND_LABELS:
            self.assertIsInstance(label, str)

    def test_bank_filter_by_clusters(self) -> None:
        clusters = state.all_clusters(self.bank)
        self.assertEqual(len(clusters), 8)

        every = state.question_ids_for_clusters(self.bank, clusters)
        self.assertEqual(every, [q.id for q in self.bank if q.tier == "filter"])

        one = state.question_ids_for_clusters(self.bank, ["cell_theory"])
        self.assertTrue(one)
        self.assertTrue(set(one).issubset(set(every)))
        for qid in one:
            question = self.bank.get(qid)
            self.assertEqual(question.cluster, "cell_theory")
            self.assertEqual(question.tier, "filter")

        self.assertEqual(state.question_ids_for_clusters(self.bank, []), [])

    def test_mock_backend_smoke(self) -> None:
        session = state.build_session(
            self.bank, ["cell_theory"], client=MockInferenceClient()
        )
        question = session.current_question()
        self.assertIsNotNone(question)
        self.assertEqual(question.tier, "filter")

        result = session.submit_answer(question.correct_answer)
        self.assertEqual(result.kind, "correct")
        self.assertEqual(result.question_id, question.id)

        wrong = session.current_question()
        self.assertIsNotNone(wrong)
        result = session.submit_answer("a completely unrelated answer")
        self.assertIn(result.kind, {"filter_missed", "filter_escalated"})

        rows = state.answer_rows(session.state.history)
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["turn"] for r in rows], [1, 2])
        self.assertEqual(rows[0]["verdict"], "correct")
        self.assertEqual(rows[1]["verdict"], "wrong")

        current, delta = state.mastery_delta(session.state.history, "cell_theory")
        self.assertIsNotNone(current)
        self.assertIsNotNone(delta)

        self.assertEqual(state.resolved_at(None), "bank exact match")
        self.assertEqual(state.resolved_at("llm"), "layer 3 · LLM classifier")

    def test_turn_slice_does_not_leak_previous_turn(self) -> None:
        """A turn's slice must stop at its own classifier run.

        A missed filter logs escalation_choice *after* its answer entry; that
        entry belongs to the filter turn, not to the Socratic turn it injects.
        """
        session = state.build_session(
            self.bank, ["cell_theory"], client=MockInferenceClient()
        )
        session.submit_answer("banana")  # wrong filter -> escalates
        first = state.last_turn_slice(session.state.history)
        self.assertIsNotNone(state.find_event(first, "escalation_choice"))

        injected = session.current_question()
        self.assertEqual(injected.tier, "socratic")

        session.submit_answer("tissues get bigger because the body makes more stuff")
        second = state.last_turn_slice(session.state.history)
        self.assertIsNone(state.find_event(second, "escalation_choice"))
        self.assertEqual(second[0].get("event"), "llm_classify")
        self.assertIsNotNone(state.find_event(second, "answer"))


if __name__ == "__main__":
    unittest.main()
