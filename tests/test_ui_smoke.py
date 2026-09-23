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
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from socratic_core.mock_client import MockInferenceClient  # noqa: E402
from socratic_core.question_bank import load_question_bank  # noqa: E402
from socratic_core.state_machine import ENV_SESSIONS_DIR, SocraticSession  # noqa: E402
from ui import state, student_view  # noqa: E402


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

    def test_build_presets_socratic_has_three_to_four(self) -> None:
        socratic = [q for q in self.bank if q.tier == "socratic"]
        self.assertTrue(socratic)
        for question in socratic:
            presets = student_view.build_presets(question)
            with self.subTest(question=question.id):
                self.assertTrue(3 <= len(presets) <= 4, presets)
                self.assertEqual(presets[0], student_view.PRESET_GIVE_UP)
                self.assertEqual(presets[-1], question.correct_answer)
                self.assertEqual(presets[1], question.misconceptions[0].wrong_answer)
                self.assertEqual(len(set(presets)), len(presets))

    def test_build_presets_filter_has_two(self) -> None:
        filters = [q for q in self.bank if q.tier == "filter"]
        self.assertTrue(filters)
        for question in filters:
            with self.subTest(question=question.id):
                self.assertEqual(
                    student_view.build_presets(question),
                    [student_view.PRESET_GIVE_UP, question.correct_answer],
                )

    def test_partial_preset_misses_one_key_term(self) -> None:
        """The partial preset must miss exactly one key term and reach the LLM.

        Since Phase 3 a partial lexical match is not a verdict: it falls
        through to the LLM classifier, which decides partial. Asserted against
        the state machine's own key-terms check rather than a re-implementation.
        """
        checked = 0
        for question in self.bank:
            if question.tier != "socratic":
                continue
            presets = student_view.build_presets(question)
            if len(presets) < 4:
                continue
            verdict, matched, missing = SocraticSession._classify_by_key_terms(
                question, presets[2]
            )
            with self.subTest(question=question.id):
                self.assertIsNone(verdict)  # not Case A -> the LLM classifier decides
                self.assertEqual(missing, [question.key_terms[0]])
                self.assertTrue(matched)
                self.assertFalse(question.is_correct(presets[2]))
            checked += 1
        self.assertEqual(checked, len([q for q in self.bank if q.tier == "socratic"]))


    def test_submit_keeps_answer_in_draft(self) -> None:
        """The graded answer stays in the box so it sits beside its verdict."""
        session = state.build_session(
            self.bank, ["cell_theory"], client=MockInferenceClient()
        )
        question = session.current_question()
        stub = SimpleNamespace(
            session_state=SimpleNamespace(
                sc_answer_draft=question.correct_answer,
                sc_session=session,
                sc_last_result=None,
                sc_turn_count=0,
            )
        )

        with mock.patch.object(student_view, "st", stub):
            student_view._submit()
        self.assertEqual(stub.session_state.sc_answer_draft, question.correct_answer)
        self.assertEqual(stub.session_state.sc_turn_count, 1)
        self.assertEqual(stub.session_state.sc_last_result.kind, "correct")

        # Dismissing feedback must not disturb the preserved answer.
        with mock.patch.object(student_view, "st", stub):
            student_view._dismiss_feedback()
        self.assertEqual(stub.session_state.sc_answer_draft, question.correct_answer)
        self.assertIsNone(stub.session_state.sc_last_result)

        # A preset click overwrites it; a blank submit is ignored.
        with mock.patch.object(student_view, "st", stub):
            student_view._fill_draft("No idea")
            self.assertEqual(stub.session_state.sc_answer_draft, "No idea")
            stub.session_state.sc_answer_draft = "   "
            student_view._submit()
        self.assertEqual(stub.session_state.sc_turn_count, 1)


    def test_feedback_card_shows_graded_answer(self) -> None:
        """The card is labelled from the log, so a preset click cannot desync it."""
        session = state.build_session(
            self.bank, ["cell_theory"], client=MockInferenceClient()
        )
        self.assertIsNone(student_view.graded_answer(session))

        submitted = "  banana  "
        session.submit_answer(submitted)
        # The state machine logs the stripped answer; that is what was graded.
        self.assertEqual(student_view.graded_answer(session), submitted.strip())

        # Overwriting the draft (preset click) must not change the label.
        stub = SimpleNamespace(session_state=SimpleNamespace(sc_answer_draft=""))
        with mock.patch.object(student_view, "st", stub):
            student_view._fill_draft("It splits into new cells")
        self.assertEqual(stub.session_state.sc_answer_draft, "It splits into new cells")
        self.assertEqual(student_view.graded_answer(session), "banana")

        # A second turn relabels the card.
        session.submit_answer("Rudolf Virchow")
        self.assertEqual(student_view.graded_answer(session), "Rudolf Virchow")


if __name__ == "__main__":
    unittest.main()
