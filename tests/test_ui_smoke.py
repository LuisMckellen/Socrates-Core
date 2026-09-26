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

from socratic_core.classifier_llm import make_classifier_fn, make_verify_fn  # noqa: E402
from socratic_core.mock_client import MockInferenceClient  # noqa: E402
from socratic_core.question_bank import load_question_bank  # noqa: E402
from socratic_core.state_machine import ENV_SESSIONS_DIR, SocraticSession, TurnResult  # noqa: E402
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

    def test_build_presets_socratic_has_three(self) -> None:
        socratic = [q for q in self.bank if q.tier == "socratic" and q.id != student_view.CELL_THEORY_L1]
        self.assertTrue(socratic)
        for question in socratic:
            with self.subTest(question=question.id):
                self.assertEqual(
                    student_view.build_presets(question),
                    [
                        student_view.PRESET_GIVE_UP,
                        question.misconceptions[0].wrong_answer,
                        question.correct_answer,
                    ],
                )

    def test_build_presets_cell_theory_L1(self) -> None:
        question = self.bank.get(student_view.CELL_THEORY_L1)
        self.assertEqual(
            student_view.build_presets(question),
            [
                "No idea",
                "Wounds heal because damaged tissue cells absorb surrounding nutrients and swell "
                "in size until the empty space in the tissue is filled.",
                "Cells divide because they don't divide",
                "Pre-existing cells undergo division to form new tissue",
            ],
        )
        self.assertEqual(question.misconceptions[0].id, "ct_hypertrophy")
        # The contradiction is a manual string, not a bank misconception.
        self.assertNotIn(
            student_view.CELL_THEORY_L1_CONTRADICTION, [m.wrong_answer for m in question.misconceptions]
        )
        # The correct preset is Case A (both key terms), not a bank string: verify_fn runs.
        correct = student_view.CELL_THEORY_L1_CORRECT
        self.assertFalse(question.is_correct(correct))
        self.assertEqual(SocraticSession._classify_by_key_terms(question, correct)[0], "correct")

    def test_cell_theory_L1_presets_mock_mode(self) -> None:
        """Each L1 preset, from a fresh first attempt, on the mock backend."""
        client = state.make_client(state.BACKEND_MOCK)
        question = self.bank.get(student_view.CELL_THEORY_L1)
        expected = [
            ("low_effort", "hint", student_view.MODE_PROBE),
            ("wrong", "hint", student_view.MODE_PROBE),
            ("wrong", "hint", student_view.MODE_PROBE),
            ("correct", "correct", student_view.MODE_FINISHED),  # the only question: advancing ends it
        ]
        for preset, (verdict, kind, mode) in zip(student_view.build_presets(question), expected):
            session = SocraticSession(
                self.bank, question_ids=[question.id], classifier_fn=make_classifier_fn(client),
                verify_fn=make_verify_fn(client),
            )
            result = session.submit_answer(preset)
            with self.subTest(preset=preset):
                self.assertEqual(state.last_answer_event(session.state.history)["verdict"], verdict)
                self.assertEqual(result.kind, kind)
                self.assertEqual(student_view.render_mode(result), mode)

    def test_render_mode(self) -> None:
        def result(kind: str, finished: bool = False) -> TurnResult:
            return TurnResult(kind=kind, question_id="q", attempt=1, message="m", session_finished=finished)

        self.assertEqual(student_view.render_mode(None), "bank")
        self.assertEqual(student_view.render_mode(result("hint")), "probe")
        for kind in ("correct", "escalated", "filter_escalated", "filter_missed"):
            with self.subTest(kind=kind):
                self.assertEqual(student_view.render_mode(result(kind)), "escalate")
                # The final turn carries an advancing kind too; finished wins.
                self.assertEqual(student_view.render_mode(result(kind, finished=True)), "finished")
        self.assertEqual(student_view.submit_label("probe"), "Revise")
        for mode in ("bank", "escalate"):
            self.assertEqual(student_view.submit_label(mode), "Submit Answer")

    def test_build_presets_filter_has_two(self) -> None:
        filters = [q for q in self.bank if q.tier == "filter"]
        self.assertTrue(filters)
        for question in filters:
            with self.subTest(question=question.id):
                self.assertEqual(
                    student_view.build_presets(question),
                    [student_view.PRESET_GIVE_UP, question.correct_answer],
                )

    def test_submit_clears_draft_on_advance(self) -> None:
        """The graded answer stays beside its verdict; "Next question" clears it."""
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

        # "Next question" after an advance clears it: it answered the old question.
        with mock.patch.object(student_view, "st", stub):
            student_view._dismiss_feedback()
        self.assertEqual(stub.session_state.sc_answer_draft, "")
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

    def test_make_client_raises_on_unknown_label(self) -> None:
        # The exact label a pre-rename Streamlit session still held.
        with self.assertRaises(ValueError) as cm:
            state.make_client("Local CPU (~6.5s)")
        self.assertIn("Local CPU (~6.5s)", str(cm.exception))
        with self.assertRaises(ValueError):
            state.make_client("")
        # Known labels route explicitly (Local and Groq need a model / key, so only Mock is built here).
        self.assertIsInstance(state.make_client(state.BACKEND_MOCK), MockInferenceClient)

    def test_llm_called_true_on_groq_fallback(self) -> None:
        self.assertTrue(state.llm_called("llm_groq_fallback"))
        self.assertTrue(state.llm_called("llm"))
        for source in (None, "", "behavioural", "key_terms", "key_terms_verified"):
            with self.subTest(source=source):
                self.assertFalse(state.llm_called(source))

    def test_resolved_at_handles_key_terms_verified(self) -> None:
        self.assertEqual(state.resolved_at("key_terms_verified"), "layer 2 · key terms (verified)")
        self.assertNotEqual(state.resolved_at("key_terms_verified"), "key_terms_verified")  # not the raw passthrough
        self.assertEqual(state.resolved_at("key_terms"), "layer 2 · key terms")

    def test_answer_rows_phase4_columns(self) -> None:
        long_hint = "x" * 60
        history = [
            {"event": "answer", "attempt_number": 1, "matched_bank_id": None, "elapsed_ms": 0, "hint_text": ""},
            {"event": "answer", "attempt_number": 2, "matched_bank_id": "none", "elapsed_ms": 812, "hint_text": long_hint},
            {"event": "answer", "attempt_number": 1, "matched_bank_id": "ct_spontaneous_generation", "elapsed_ms": 9, "hint_text": "Short?"},
            {"event": "answer"},  # a log from before these fields existed
        ]
        rows = state.answer_rows(history)
        self.assertEqual([r["matched_bank_id"] for r in rows], ["—", "no match", "ct_spontaneous_generation", "—"])
        self.assertEqual([r["attempt_number"] for r in rows], [1, 2, 1, None])
        self.assertEqual([r["elapsed_ms"] for r in rows], [0, 812, 9, None])
        self.assertEqual([r["hint_text"] for r in rows], ["", "x" * 40 + "…", "Short?", ""])

if __name__ == "__main__":
    unittest.main()
