"""Offline tests for the two-tier state machine (filter probes + Socratic pipeline).

Run from the project root:  python -m unittest discover -s tests -v
No model, no NPU, no network. Expected mastery values are derived from the
constants in mastery.py; this file must not restate any of them as a literal.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from socratic_core import classifier_behavioral  # noqa: E402
from socratic_core.classifier_llm import make_classifier_fn, make_verify_fn  # noqa: E402
from socratic_core.mock_client import CLASSIFIER_PROMPT_MARKER, MockInferenceClient  # noqa: E402
from socratic_core.disengagement import DISENGAGEMENT_TOKENS  # noqa: E402
from socratic_core.mastery import (  # noqa: E402
    FILTER_CORRECT_DELTA,
    FILTER_WRONG_DELTA,
    INITIAL_MASTERY,
    MASTERY_CEILING,
    MASTERY_FLOOR,
    SOCRATIC_CORRECT_DELTA,
    SOCRATIC_PARTIAL_DELTA,
    SOCRATIC_WRONG_DELTA,
)
from socratic_core.noise import NOISE_TOLERANCE_PREFIX  # noqa: E402
from socratic_core.question_bank import Question, QuestionBank, QuestionBankError, load_question_bank  # noqa: E402
from socratic_core.state_machine import SocraticSession  # noqa: E402

F_CELL = {
    "id": "f_001",
    "tier": "filter",
    "cluster": "cell_theory",
    "topic": "Cell theory",
    "question_text": "What is the basic unit of life?",
    "correct_answer": "the cell",
    "accepted_variants": ["cell", "cells"],
    "key_terms": [],
    "min_words": 1,
    "misconceptions": [],
    "answer_explanation": "",
    "fallback_hint": "",
}
F_ATP = {
    "id": "f_002",
    "tier": "filter",
    "cluster": "organelles",
    "topic": "Organelles",
    "question_text": "Which organelle produces ATP?",
    "correct_answer": "Mitochondria",
    "accepted_variants": ["Mitochondrion"],
    "key_terms": [],
    "min_words": 1,
    "misconceptions": [],
    "answer_explanation": "",
    "fallback_hint": "",
}
S_CELL = {
    "id": "s_001",
    "tier": "socratic",
    "cluster": "cell_theory",
    "topic": "Cell theory",
    "level": 1,
    "question_text": "According to cell theory, what is the basic structural and functional unit of all living organisms?",
    "correct_answer": "the cell",
    "accepted_variants": ["cell", "cells"],
    "key_terms": ["cell", "division"],
    "min_words": 3,
    "misconceptions": [
        {"wrong_answer": "the atom is the unit", "error_type": "logic_error", "explanation": "Atoms are not alive."},
        {"wrong_answer": "the organ is the unit", "error_type": "logic_error", "explanation": "Organs are made of cells."},
        {"wrong_answer": "the smallest living thing", "error_type": "logic_error", "explanation": "Name it: the cell."},
    ],
    "answer_explanation": "Every living thing is made of one or more cells.",
    "fallback_hint": "What is the smallest part of your body that is still alive on its own?",
}
FULL_BANK = {"questions": [F_CELL, S_CELL, F_ATP]}
FILTERS_ONLY_BANK = {"questions": [F_CELL, F_ATP]}


class StateMachineTierTests(unittest.TestCase):
    def setUp(self):
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.sessions_dir = tmp / "sessions"
        full = tmp / "full.json"
        full.write_text(json.dumps(FULL_BANK), encoding="utf-8")
        filters_only = tmp / "filters_only.json"
        filters_only.write_text(json.dumps(FILTERS_ONLY_BANK), encoding="utf-8")
        self.full = load_question_bank(full)
        self.filters_only = load_question_bank(filters_only)
        # Spies for the LLM slots; filter-tier tests assert these are never reached.
        self.llm = Mock(name="classifier_fn")
        self.hint = Mock(name="hint_fn")

    def test_filter_correct_does_not_call_llm(self):
        s = SocraticSession(self.full, classifier_fn=self.llm, hint_fn=self.hint, sessions_dir=self.sessions_dir)
        self.assertEqual(s.state.question_ids, ["f_001", "f_002"])  # filters only by default
        r = s.submit_answer("The Cell.")
        self.assertEqual(r.kind, "correct")
        self.assertEqual(s.state.solved_question_ids, ["f_001"])
        self.assertEqual(s.current_question().id, "f_002")
        self.assertAlmostEqual(s.state.mastery["cell_theory"], INITIAL_MASTERY + FILTER_CORRECT_DELTA)
        self.llm.assert_not_called()
        self.hint.assert_not_called()

    def test_filter_wrong_escalates_when_candidate_exists(self):
        s = SocraticSession(self.full, classifier_fn=self.llm, hint_fn=self.hint, sessions_dir=self.sessions_dir)
        r = s.submit_answer("the atom")
        self.assertEqual(r.kind, "filter_escalated")
        self.assertEqual(s.current_question().id, "s_001")
        self.assertEqual(s.state.question_ids, ["f_001", "s_001", "f_002"])
        self.assertAlmostEqual(s.state.mastery["cell_theory"], INITIAL_MASTERY + FILTER_WRONG_DELTA)
        choice = [h for h in s.state.history if h["event"] == "escalation_choice"]
        self.assertEqual(len(choice), 1)
        self.assertEqual(choice[0]["from_filter_id"], "f_001")
        self.assertEqual(choice[0]["to_question_id"], "s_001")
        self.assertEqual(choice[0]["reason"], "cluster_match")
        self.assertNotIn("topic_gap", [h["event"] for h in s.state.history])
        self.assertEqual(s.state.solved_question_ids, [])
        self.assertEqual(s.state.stuck_question_ids, [])
        self.llm.assert_not_called()
        self.hint.assert_not_called()

    def test_filter_wrong_logs_gap_when_no_candidate(self):
        s = SocraticSession(self.filters_only, classifier_fn=self.llm, hint_fn=self.hint, sessions_dir=self.sessions_dir)
        r = s.submit_answer("the atom")
        self.assertEqual(r.kind, "filter_missed")
        self.assertEqual(s.current_question().id, "f_002")
        gaps = [h for h in s.state.history if h["event"] == "topic_gap"]
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["cluster"], "cell_theory")
        self.assertEqual(gaps[0]["reason"], "no_socratic_candidate")
        self.assertNotIn("escalation_choice", [h["event"] for h in s.state.history])
        self.llm.assert_not_called()
        self.hint.assert_not_called()

    def test_filter_wrong_logs_gap_when_mastery_above_threshold(self):
        s = SocraticSession(self.full, classifier_fn=self.llm, hint_fn=self.hint, sessions_dir=self.sessions_dir)
        s.state.mastery["cell_theory"] = MASTERY_CEILING
        r = s.submit_answer("the atom")
        self.assertEqual(r.kind, "filter_missed")
        self.assertEqual(s.current_question().id, "f_002")  # s_001 was NOT injected
        gaps = [h for h in s.state.history if h["event"] == "topic_gap"]
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["reason"], "mastery_above_threshold")
        self.assertAlmostEqual(s.state.mastery["cell_theory"], MASTERY_CEILING + FILTER_WRONG_DELTA)
        self.assertNotIn("escalation_choice", [h["event"] for h in s.state.history])
        self.llm.assert_not_called()
        self.hint.assert_not_called()

    def test_socratic_tier_uses_full_pipeline(self):
        # S_CELL key_terms = ["cell", "division"]: behavioural -> key_terms (A/C) -> LLM.
        classified: list[str] = []
        hinted: list[str] = []
        labels = iter(["partial", "wording_error"])

        def fake_llm(question, answer):
            classified.append(answer)
            return {"error_type": next(labels), "confidence": 0.9, "reasoning": "test"}

        def fake_hint(question, answer, error_type):
            hinted.append(error_type)
            return "Is an atom alive on its own?"

        s = SocraticSession(
            self.full, question_ids=["s_001"], classifier_fn=fake_llm, hint_fn=fake_hint, sessions_dir=self.sessions_dir
        )
        r = s.submit_answer("idk")  # behavioural layer
        self.assertEqual((r.kind, r.error_type, r.error_source), ("hint", "low_effort", "behavioural"))
        self.assertEqual(classified, [])  # the behavioural turn never reached the LLM
        r = s.submit_answer("cells are the unit of life")  # "cell" hit, "division" missing -> LLM says partial
        self.assertEqual((r.error_type, r.error_source), ("logic_error", "llm"))
        self.assertEqual(len(classified), 1)
        self.assertEqual([h for h in s.state.history if h["event"] == "answer"][-1]["verdict"], "partial")
        r = s.submit_answer("the cell")
        self.assertEqual(r.kind, "correct")
        self.assertAlmostEqual(
            s.state.mastery["cell_theory"],
            INITIAL_MASTERY + SOCRATIC_PARTIAL_DELTA + SOCRATIC_CORRECT_DELTA,
        )

        s2 = SocraticSession(
            self.full, question_ids=["s_001"], classifier_fn=fake_llm, hint_fn=fake_hint, sessions_dir=self.sessions_dir
        )
        r = s2.submit_answer("the organ system in the body")  # zero key_terms -> LLM layer + hint
        self.assertEqual((r.error_type, r.error_source), ("wording_error", "llm"))
        self.assertEqual(len(classified), 2)
        self.assertTrue(classified[1].startswith(NOISE_TOLERANCE_PREFIX))
        self.assertTrue(classified[1].endswith("the organ system in the body"))
        # Every wrong turn across both sessions reached the hint generator with its layer's verdict.
        self.assertEqual(hinted, ["low_effort", "logic_error", "wording_error"])
        self.assertEqual(r.message, "Is an atom alive on its own?")

    def test_mastery_trajectory_logged(self):
        s = SocraticSession(self.full, classifier_fn=self.llm, hint_fn=self.hint, sessions_dir=self.sessions_dir)
        s.submit_answer("cell")  # f_001 correct
        s.submit_answer("nucleus")  # f_002 wrong; organelles has no Socratic entry -> global fallback to s_001
        snapshots = [h["mastery"] for h in s.state.history if h["event"] == "mastery_snapshot"]
        self.assertEqual(len(snapshots), 2)
        self.assertAlmostEqual(snapshots[0]["cell_theory"], INITIAL_MASTERY + FILTER_CORRECT_DELTA)
        self.assertNotIn("organelles", snapshots[0])
        self.assertAlmostEqual(snapshots[1]["cell_theory"], INITIAL_MASTERY + FILTER_CORRECT_DELTA)
        self.assertAlmostEqual(snapshots[1]["organelles"], INITIAL_MASTERY + FILTER_WRONG_DELTA)
        choice = [h for h in s.state.history if h["event"] == "escalation_choice"][0]
        self.assertEqual((choice["from_filter_id"], choice["to_question_id"], choice["reason"]), ("f_002", "s_001", "global_fallback"))
        log = json.loads(s.log_path.read_text(encoding="utf-8"))
        self.assertEqual(log["mastery"], s.state.mastery)

    def test_filter_answers_skip_behavioral_layer(self):
        with (
            patch.object(classifier_behavioral, "explain", side_effect=AssertionError("behavioural layer ran on a filter")) as explain,
            patch.object(classifier_behavioral, "classify_behavioural", side_effect=AssertionError("behavioural layer ran on a filter")) as classify,
        ):
            s = SocraticSession(self.filters_only, classifier_fn=self.llm, hint_fn=self.hint, sessions_dir=self.sessions_dir)
            s.submit_answer("")  # empty
            self.assertEqual(s.current_question().id, "f_002")
            s.submit_answer("asdf qwerty zzz")  # gibberish, off-topic, long enough to pass min_words
            self.assertTrue(s.state.finished)

            s = SocraticSession(self.filters_only, classifier_fn=self.llm, hint_fn=self.hint, sessions_dir=self.sessions_dir)
            s.submit_answer("cell")  # correct
            s.submit_answer("mitochondria")  # correct
            self.assertEqual(s.state.solved_question_ids, ["f_001", "f_002"])
        explain.assert_not_called()
        classify.assert_not_called()
        self.llm.assert_not_called()
        self.hint.assert_not_called()

    def test_filter_idk_escalates(self):
        s = SocraticSession(self.full, classifier_fn=self.llm, hint_fn=self.hint, sessions_dir=self.sessions_dir)
        r = s.submit_answer("idk")
        self.assertEqual(r.kind, "filter_escalated")
        self.assertIsNone(r.error_type)
        self.assertIsNone(r.error_source)
        self.assertEqual(s.current_question().id, "s_001")
        self.assertIn("escalation_choice", [h["event"] for h in s.state.history])
        self.assertNotIn("low_effort", json.dumps(s.state.history))
        answer_entry = [h for h in s.state.history if h["event"] == "answer"][0]
        self.assertFalse(answer_entry["correct"])
        self.assertNotIn("error_type", answer_entry)
        self.llm.assert_not_called()
        self.hint.assert_not_called()

    # -- Change 4/5: key_terms Case A/B/C, synonyms, noise tolerance ---------

    def test_socratic_correct_by_key_terms(self):
        # Not an exact string match, but both key_terms ("cell", "division") are present.
        s = SocraticSession(
            self.full, question_ids=["s_001"], classifier_fn=self.llm, hint_fn=self.hint,
            verify_fn=make_verify_fn(MockInferenceClient()), sessions_dir=self.sessions_dir,
        )
        r = s.submit_answer("cells form through division of other cells")
        self.assertEqual(r.kind, "correct")
        answer_entry = [h for h in s.state.history if h["event"] == "answer"][0]
        self.assertEqual(answer_entry["kind"], "socratic_correct")
        self.assertEqual(answer_entry["verdict"], "correct")
        self.assertEqual(answer_entry["error_source"], "key_terms_verified")
        self.assertIs(answer_entry["case_a_verified"], True)
        self.assertEqual(set(answer_entry["key_terms_matched"]), {"cell", "division"})
        self.assertEqual(answer_entry["key_terms_missing"], [])
        self.llm.assert_not_called()

    # -- Phase 3: partial is an LLM verdict (Case B is gone) ------------------

    @staticmethod
    def _partial_llm(question, answer):
        return {"error_type": "partial", "confidence": 0.9, "reasoning": "half of it"}

    def test_llm_returns_partial_label(self):
        llm = Mock(side_effect=self._partial_llm)
        s = SocraticSession(self.full, question_ids=["s_001"], classifier_fn=llm, hint_fn=lambda q, a, e: "hint?", sessions_dir=self.sessions_dir)
        r = s.submit_answer("cells are the smallest unit")  # "cell" hit, "division" missing -> LLM, not key_terms
        llm.assert_called_once()
        self.assertEqual((r.kind, r.error_type, r.error_source), ("hint", "logic_error", "llm"))
        answer_entry = [h for h in s.state.history if h["event"] == "answer"][0]
        self.assertEqual((answer_entry["verdict"], answer_entry["kind"]), ("partial", "socratic_partial"))
        self.assertEqual((answer_entry["key_terms_matched"], answer_entry["key_terms_missing"]), (["cell"], ["division"]))
        classified = [h for h in s.state.history if h["event"] == "llm_classify"][0]
        self.assertEqual((classified["llm_label"], classified["error_type"], classified["verdict"]), ("partial", "logic_error", "partial"))

    def test_partial_uses_socratic_partial_delta(self):
        s = SocraticSession(self.full, question_ids=["s_001"], classifier_fn=self._partial_llm, hint_fn=lambda q, a, e: "hint?", sessions_dir=self.sessions_dir)
        s.submit_answer("cells are the smallest unit")
        self.assertAlmostEqual(s.state.mastery["cell_theory"], INITIAL_MASTERY + SOCRATIC_PARTIAL_DELTA)

    def test_partial_hint_targets_missing_terms(self):
        seen: list[tuple[str, object]] = []

        def hint(question, answer, error_type, *, missing_terms=None):
            seen.append((error_type, missing_terms))
            return "What must the cells do?"

        s = SocraticSession(self.full, question_ids=["s_001"], classifier_fn=self._partial_llm, hint_fn=hint, sessions_dir=self.sessions_dir)
        s.submit_answer("cells are the smallest unit")
        self.assertEqual(seen, [("logic_error", ["division"])])

        # Case A verified NO, then the LLM said partial: nothing lexical is
        # missing, so no missing_terms -> the logic_error hint strategy.
        seen.clear()
        s = SocraticSession(
            self.full, question_ids=["s_001"], classifier_fn=self._partial_llm, hint_fn=hint,
            verify_fn=lambda q, a: False, sessions_dir=self.sessions_dir,
        )
        s.submit_answer("cells form through division of other cells")
        self.assertEqual(seen, [("logic_error", None)])

    def test_partial_counts_toward_reveal(self):
        # CORRECTIONS.md #9: a third partial attempt reveals, like any other.
        s = SocraticSession(self.full, question_ids=["s_001"], classifier_fn=self._partial_llm, hint_fn=lambda q, a, e: "hint?", sessions_dir=self.sessions_dir)
        kinds = [s.submit_answer("cells are the smallest unit").kind for _ in range(3)]
        self.assertEqual(kinds, ["hint", "hint", "escalated"])
        self.assertEqual(s.state.stuck_question_ids, ["s_001"])

    def test_no_case_b_remains(self):
        source = (ROOT / "socratic_core" / "state_machine.py").read_text(encoding="utf-8")
        for needle in ('kt_verdict == "partial"', "partial key-term", 'return "partial"', "Case A/B/C"):
            self.assertNotIn(needle, source)
        for answer in ("cells are the smallest unit", "cells divide skibidi rizz"):  # partial lexical matches
            self.assertIsNone(SocraticSession._classify_by_key_terms(self.full.get("s_001"), answer)[0])

    def test_socratic_wrong_costs_mastery_every_attempt_clamped_at_floor(self):
        def fake_llm(question, answer):
            return {"error_type": "logic_error", "confidence": 0.9, "reasoning": "test"}

        s = SocraticSession(
            self.full, question_ids=["s_001"], classifier_fn=fake_llm, hint_fn=lambda q, a, e: "hint?",
            sessions_dir=self.sessions_dir, max_attempts=3,
        )
        answers = ["the organ system in the body", "the atom is the unit", "the tissue in the body"]  # zero key_terms
        kinds, trajectory = [], []
        for answer in answers:
            kinds.append(s.submit_answer(answer).kind)
            trajectory.append(s.state.mastery["cell_theory"])
        self.assertEqual(kinds, ["hint", "hint", "escalated"])
        expected, level = [], INITIAL_MASTERY
        for _ in answers:
            level = max(MASTERY_FLOOR, level + SOCRATIC_WRONG_DELTA)
            expected.append(level)
        for actual, want in zip(trajectory, expected):
            self.assertAlmostEqual(actual, want)
        self.assertAlmostEqual(trajectory[0], INITIAL_MASTERY + SOCRATIC_WRONG_DELTA)  # immediate, not deferred
        self.assertEqual(trajectory[-1], MASTERY_FLOOR)  # clamped, not negative

    def test_socratic_zero_key_terms_falls_through_to_llm(self):
        calls: list[str] = []

        def fake_llm(question, answer):
            calls.append(answer)
            return {"error_type": "logic_error", "confidence": 0.9, "reasoning": "test"}

        s = SocraticSession(self.full, question_ids=["s_001"], classifier_fn=fake_llm, hint_fn=lambda q, a, e: "hint?", sessions_dir=self.sessions_dir)
        r = s.submit_answer("the organ system in the body")  # neither "cell" nor "division"
        self.assertEqual(len(calls), 1)
        self.assertEqual(r.error_source, "llm")
        answer_entry = [h for h in s.state.history if h["event"] == "answer"][0]
        self.assertEqual(answer_entry["kind"], "socratic_wrong")
        self.assertEqual(answer_entry["verdict"], "wrong")

    def test_socratic_splits_matches_divide_via_synonym_map(self):
        question = Question(
            "s_syn", "Synonyms", "How do cells reproduce?", "cells divide",
            tier="socratic", cluster="synonyms", level=1, key_terms=("divide",), fallback_hint="?",
        )
        verdict, matched, missing = SocraticSession._classify_by_key_terms(question, "the cell splits apart")
        self.assertEqual(verdict, "correct")
        self.assertEqual(matched, ["divide"])
        self.assertEqual(missing, [])

    def test_noise_tolerance_prefix_in_classifier_prompt(self):
        captured: list[str] = []

        def fake_llm(question, answer):
            captured.append(answer)
            return {"error_type": "logic_error", "confidence": 0.9, "reasoning": "test"}

        s = SocraticSession(self.full, question_ids=["s_001"], classifier_fn=fake_llm, hint_fn=lambda q, a, e: "hint?", sessions_dir=self.sessions_dir)
        s.submit_answer("the organ system in the body")
        self.assertEqual(len(captured), 1)
        self.assertTrue(captured[0].startswith(NOISE_TOLERANCE_PREFIX))

    # -- Change 6: disengagement logging (Option A: never affects verdict) --

    @staticmethod
    def _logic_llm(question, answer):
        return {"error_type": "logic_error", "confidence": 0.9, "reasoning": "test"}

    def test_disengagement_flag_set_on_meme_tokens(self):
        # "cells" hits, "division" missing -> reaches the LLM since Phase 3.
        s = SocraticSession(self.full, question_ids=["s_001"], classifier_fn=self._logic_llm, hint_fn=lambda q, a, e: "hint?", sessions_dir=self.sessions_dir)
        s.submit_answer("cells divide skibidi rizz")
        answer_entry = [h for h in s.state.history if h["event"] == "answer"][0]
        self.assertTrue(answer_entry["disengagement_flag"])
        self.assertEqual(set(answer_entry["disengagement_tokens"]), {"skibidi", "rizz"})

    def test_disengagement_flag_not_set_on_clean_answer(self):
        s = SocraticSession(self.full, question_ids=["s_001"], classifier_fn=self._logic_llm, hint_fn=lambda q, a, e: "hint?", sessions_dir=self.sessions_dir)
        s.submit_answer("cells divide to form new cells")
        answer_entry = [h for h in s.state.history if h["event"] == "answer"][0]
        self.assertFalse(answer_entry["disengagement_flag"])
        self.assertEqual(answer_entry["disengagement_tokens"], [])

    def test_disengagement_does_not_affect_verdict(self):
        clean = SocraticSession(self.full, question_ids=["s_001"], classifier_fn=self.llm, hint_fn=self.hint, sessions_dir=self.sessions_dir)
        r_clean = clean.submit_answer("cells go through division here")
        memed = SocraticSession(self.full, question_ids=["s_001"], classifier_fn=self.llm, hint_fn=self.hint, sessions_dir=self.sessions_dir)
        r_memed = memed.submit_answer("cells go through division here skibidi rizz")
        self.assertEqual(r_clean.kind, r_memed.kind)
        clean_entry = [h for h in clean.state.history if h["event"] == "answer"][0]
        memed_entry = [h for h in memed.state.history if h["event"] == "answer"][0]
        self.assertEqual(clean_entry["verdict"], memed_entry["verdict"])
        self.assertFalse(clean_entry["disengagement_flag"])
        self.assertTrue(memed_entry["disengagement_flag"])

    def test_meme_answer_with_concept_still_scores_correct(self):
        question = Question(
            "s_test_L1", "Test", "q?", "alpha beta",
            tier="socratic", cluster="test", level=1, key_terms=("alpha", "beta"),
            min_words=1, fallback_hint="?",
        )
        verdict, matched, missing = SocraticSession._classify_by_key_terms(question, "alpha beta skibidi rizz")
        self.assertEqual(verdict, "correct")
        self.assertEqual(missing, [])

    def test_meme_answer_without_concept_still_scores_wrong(self):
        question = Question(
            "s_test_L1", "Test", "q?", "alpha beta",
            tier="socratic", cluster="test", level=1, key_terms=("alpha", "beta"),
            min_words=1, fallback_hint="?",
        )
        verdict, matched, missing = SocraticSession._classify_by_key_terms(question, "skibidi rizz")
        self.assertNotEqual(verdict, "correct")
        self.assertEqual(matched, [])

    def test_meme_without_concept_scores_wrong_not_low_effort(self):
        # Single-token answer bypasses the behavioural off-topic rule
        # (OFF_TOPIC_MIN_CONTENT_WORDS=2) so the pipeline reaches Case C.
        # Property under test: a disengagement token on its own does NOT
        # alter the verdict. Under Option B disengagement only decides when
        # it is paired with an off-topic answer -- see the sibling test
        # test_off_topic_plus_meme_is_low_effort for that path.
        question = Question(
            "s_test_L1", "Test", "What are alpha and beta?", "alpha beta",
            accepted_variants=("alpha and beta",),
            tier="socratic", cluster="test", level=1, key_terms=("alpha", "beta"),
            min_words=1, fallback_hint="?",
        )
        calls: list[str] = []

        def fake_llm(q, answer):
            calls.append(answer)
            return {"error_type": "logic_error", "confidence": 0.9, "reasoning": "test"}

        s = SocraticSession(
            QuestionBank([question]), classifier_fn=fake_llm, hint_fn=lambda q, a, e: "hint?",
            sessions_dir=self.sessions_dir,
        )
        s.submit_answer("skibidi")
        answer_entry = [h for h in s.state.history if h["event"] == "answer"][0]
        self.assertEqual(answer_entry["verdict"], "wrong")
        self.assertEqual(answer_entry["error_source"], "llm")
        self.assertEqual(answer_entry["error_type"], "logic_error")
        self.assertTrue(answer_entry["disengagement_flag"])
        self.assertIn("skibidi", answer_entry["disengagement_tokens"])
        self.assertEqual(len(calls), 1)

    def test_off_topic_plus_meme_is_low_effort(self):
        """Option B's new path: off-topic AND a disengagement token decide."""
        question = Question(
            "s_test_L1", "Test", "What are alpha and beta?", "alpha beta",
            accepted_variants=("alpha and beta",),
            tier="socratic", cluster="test", level=1, key_terms=("alpha", "beta"),
            min_words=1, fallback_hint="?",
        )
        calls: list[str] = []

        def fake_llm(q, answer):
            calls.append(answer)
            return {"error_type": "logic_error", "confidence": 0.9, "reasoning": "test"}

        s = SocraticSession(
            QuestionBank([question]), classifier_fn=fake_llm, hint_fn=lambda q, a, e: "hint?",
            sessions_dir=self.sessions_dir,
        )
        r = s.submit_answer("skibidi my favourite football team won yesterday")
        self.assertEqual((r.error_type, r.error_source), ("low_effort", "behavioural"))
        answer_entry = [h for h in s.state.history if h["event"] == "answer"][0]
        self.assertEqual(answer_entry["verdict"], "low_effort")
        self.assertTrue(answer_entry["disengagement_flag"])
        self.assertEqual(calls, [])  # decided at the gate, never reached the LLM

    def test_off_topic_natural_language_reaches_llm(self):
        """T1 at pipeline level: a paraphrase must not die at the gate."""
        question = Question(
            "s_test_L1", "Test", "What are alpha and beta?", "alpha beta",
            accepted_variants=("alpha and beta",),
            tier="socratic", cluster="test", level=1, key_terms=("alpha", "beta"),
            min_words=1, fallback_hint="?",
        )
        calls: list[str] = []

        def fake_llm(q, answer):
            calls.append(answer)
            return {"error_type": "logic_error", "confidence": 0.9, "reasoning": "test"}

        s = SocraticSession(
            QuestionBank([question]), classifier_fn=fake_llm, hint_fn=lambda q, a, e: "hint?",
            sessions_dir=self.sessions_dir,
        )
        r = s.submit_answer("my favourite football team won yesterday")
        self.assertEqual((r.error_type, r.error_source), ("logic_error", "llm"))
        answer_entry = [h for h in s.state.history if h["event"] == "answer"][0]
        self.assertEqual(answer_entry["verdict"], "wrong")
        self.assertFalse(answer_entry["disengagement_flag"])
        self.assertEqual(len(calls), 1)


class BankValidationTests(unittest.TestCase):
    """Each test breaks exactly one rule in an otherwise valid fixture."""

    def setUp(self):
        self.path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "bank.json"

    def test_rejects_wrong_id_prefix(self):
        self.path.write_text(json.dumps({"questions": [{**S_CELL, "id": "f_099"}]}), encoding="utf-8")
        with self.assertRaises(QuestionBankError) as cm:
            load_question_bank(self.path)
        self.assertEqual(str(cm.exception), "question 'f_099': socratic id must start with 's_' (got 'f_099')")

        self.path.write_text(json.dumps({"questions": [{**F_CELL, "id": "bio_001"}]}), encoding="utf-8")
        with self.assertRaises(QuestionBankError) as cm:
            load_question_bank(self.path)
        self.assertEqual(str(cm.exception), "question 'bio_001': filter id must start with 'f_' (got 'bio_001')")

    def test_rejects_empty_accepted_variants(self):
        self.path.write_text(json.dumps({"questions": [{**F_CELL, "accepted_variants": []}]}), encoding="utf-8")
        with self.assertRaises(QuestionBankError) as cm:
            load_question_bank(self.path)
        self.assertEqual(str(cm.exception), "question 'f_001': accepted_variants must contain at least one entry")

    def test_rejects_socratic_key_terms_out_of_range(self):
        self.path.write_text(json.dumps({"questions": [{**S_CELL, "key_terms": ["a", "b", "c", "d", "e"]}]}), encoding="utf-8")
        with self.assertRaises(QuestionBankError) as cm:
            load_question_bank(self.path)
        self.assertEqual(str(cm.exception), "question 's_001': socratic key_terms must have 1–4 entries (got 5)")

        self.path.write_text(json.dumps({"questions": [{**S_CELL, "key_terms": []}]}), encoding="utf-8")
        with self.assertRaises(QuestionBankError) as cm:
            load_question_bank(self.path)
        self.assertEqual(str(cm.exception), "question 's_001': socratic key_terms must have 1–4 entries (got 0)")

    def test_rejects_min_words_out_of_range(self):
        self.path.write_text(json.dumps({"questions": [{**F_CELL, "min_words": 3}]}), encoding="utf-8")
        with self.assertRaises(QuestionBankError) as cm:
            load_question_bank(self.path)
        self.assertEqual(str(cm.exception), "question 'f_001': filter min_words must be in [1, 2] (got 3)")

        self.path.write_text(json.dumps({"questions": [{**S_CELL, "min_words": 2}]}), encoding="utf-8")
        with self.assertRaises(QuestionBankError) as cm:
            load_question_bank(self.path)
        self.assertEqual(str(cm.exception), "question 's_001': socratic min_words must be in [3, 8] (got 2)")

    def test_rejects_malformed_cluster(self):
        self.path.write_text(json.dumps({"questions": [{**F_CELL, "cluster": "Cell Theory"}]}), encoding="utf-8")
        with self.assertRaises(QuestionBankError) as cm:
            load_question_bank(self.path)
        self.assertEqual(str(cm.exception), "question 'f_001': cluster must be lowercase snake_case (got 'Cell Theory')")

    def test_rejects_duplicate_ids(self):
        self.path.write_text(json.dumps({"questions": [F_CELL, F_ATP, F_CELL]}), encoding="utf-8")
        with self.assertRaises(QuestionBankError) as cm:
            load_question_bank(self.path)
        self.assertEqual(str(cm.exception), "duplicate id 'f_001' found at indices 0 and 2")

    # -- Change 1: CHECK 7, level -------------------------------------------

    def test_rejects_socratic_missing_level(self):
        bad = {k: v for k, v in S_CELL.items() if k != "level"}
        self.path.write_text(json.dumps({"questions": [bad]}), encoding="utf-8")
        with self.assertRaises(QuestionBankError) as cm:
            load_question_bank(self.path)
        self.assertIn("requires level in [1, 2]", str(cm.exception))

    def test_rejects_socratic_invalid_level_value(self):
        self.path.write_text(json.dumps({"questions": [{**S_CELL, "level": 3}]}), encoding="utf-8")
        with self.assertRaises(QuestionBankError) as cm:
            load_question_bank(self.path)
        self.assertIn("requires level in [1, 2]", str(cm.exception))

    def test_rejects_filter_with_level_field(self):
        self.path.write_text(json.dumps({"questions": [{**F_CELL, "level": 1}]}), encoding="utf-8")
        with self.assertRaises(QuestionBankError) as cm:
            load_question_bank(self.path)
        self.assertEqual(str(cm.exception), "question 'f_001': tier 'filter' must not have a level field")

    # -- Change 2/9: bank misconceptions are logic_error only ---------------

    def test_rejects_socratic_with_wording_error_in_bank(self):
        bad_miscs = [{**S_CELL["misconceptions"][0], "error_type": "wording_error"}]
        self.path.write_text(json.dumps({"questions": [{**S_CELL, "misconceptions": bad_miscs}]}), encoding="utf-8")
        with self.assertRaises(QuestionBankError) as cm:
            load_question_bank(self.path)
        self.assertIn("not in ['logic_error']", str(cm.exception))

    # -- Change 3: CHECK 8, relaxed misconception count (1-3) ----------------

    def test_accepts_socratic_with_1_misconception(self):
        bank = load_question_bank(
            self._write({**S_CELL, "misconceptions": S_CELL["misconceptions"][:1]})
        )
        self.assertEqual(len(bank.get("s_001").misconceptions), 1)

    def test_accepts_socratic_with_2_misconceptions(self):
        bank = load_question_bank(
            self._write({**S_CELL, "misconceptions": S_CELL["misconceptions"][:2]})
        )
        self.assertEqual(len(bank.get("s_001").misconceptions), 2)

    def test_rejects_socratic_with_4_misconceptions(self):
        four = list(S_CELL["misconceptions"]) + [S_CELL["misconceptions"][0]]
        self.path.write_text(json.dumps({"questions": [{**S_CELL, "misconceptions": four}]}), encoding="utf-8")
        with self.assertRaises(QuestionBankError) as cm:
            load_question_bank(self.path)
        self.assertEqual(
            str(cm.exception), "question 's_001': tier 'socratic' must have 1-3 misconceptions (found 4)"
        )

    def _write(self, question: dict) -> Path:
        self.path.write_text(json.dumps({"questions": [question]}), encoding="utf-8")
        return self.path


class _ScriptedClient(MockInferenceClient):
    """Classifier client that returns ``text`` verbatim (hints still canned)."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text

    def _canned_text(self, prompt: str) -> str:
        if CLASSIFIER_PROMPT_MARKER in prompt:
            return self.text
        return super()._canned_text(prompt)


class LLMVerdictTests(unittest.TestCase):
    """LLM (Case C) labels -> verdicts, against the live bank's s_cell_theory_L1."""

    NATURAL = "The body makes more of itself when you get hurt"
    # Paraphrase of the bank's first misconception; avoids every key term/synonym.
    PARAPHRASE = "When you get cut the hurt cells soak up food from around them and get bigger until the gap is filled"

    def setUp(self):
        self.sessions_dir = Path(self.enterContext(tempfile.TemporaryDirectory())) / "sessions"
        self.bank = load_question_bank(ROOT / "question_bank.json")
        self.q = self.bank.get("s_cell_theory_L1")
        self.hints: list[str] = []

    def _session(self, classifier_fn):
        def hint(question, answer, error_type):
            self.hints.append(error_type)
            return "What must the cells do?"

        return SocraticSession(
            self.bank, question_ids=["s_cell_theory_L1"], classifier_fn=classifier_fn, hint_fn=hint,
            sessions_dir=self.sessions_dir,
        )

    @staticmethod
    def _events(s, event):
        return [h for h in s.state.history if h["event"] == event]

    def test_natural_language_correct_answer_scores_correct(self):
        client = _ScriptedClient("LABEL: correct\nCONFIDENCE: 0.9\nREASONING: Existing cells divide to heal.")
        s = self._session(make_classifier_fn(client))
        r = s.submit_answer(self.NATURAL)
        self.assertEqual((r.kind, r.error_type, r.error_source), ("correct", None, "llm"))
        answer = self._events(s, "answer")[-1]
        self.assertEqual(
            (answer["verdict"], answer["kind"], answer["error_source"], answer["correct"]),
            ("correct", "socratic_correct", "llm", True),
        )
        self.assertNotIn("error_type", answer)
        self.assertAlmostEqual(s.state.mastery["cell_theory"], INITIAL_MASTERY + SOCRATIC_CORRECT_DELTA)
        self.assertEqual(s.state.solved_question_ids, ["s_cell_theory_L1"])
        self.assertEqual(self.hints, [])  # correct path: no hint
        self.assertTrue(r.session_finished)
        classified = self._events(s, "llm_classify")[-1]
        self.assertEqual((classified["llm_label"], classified["verdict"], classified["kind"]), ("correct", "correct", "socratic_correct"))
        prompt = client.call_log[0]["prompt"]
        self.assertIn(f"{NOISE_TOLERANCE_PREFIX} {self.NATURAL}", prompt)

    def test_paraphrased_misconception_is_logic_error_not_correct(self):
        mock = MockInferenceClient()
        s = self._session(make_classifier_fn(mock))
        r = s.submit_answer(self.PARAPHRASE)
        self.assertEqual((r.kind, r.error_type, r.error_source), ("hint", "logic_error", "llm"))
        answer = self._events(s, "answer")[-1]
        self.assertEqual((answer["verdict"], answer["kind"], answer["correct"]), ("wrong", "socratic_wrong", False))
        self.assertAlmostEqual(s.state.mastery["cell_theory"], INITIAL_MASTERY + SOCRATIC_WRONG_DELTA)
        # The bank string is in the prompt as few-shot context; the student sent a paraphrase.
        prompt = mock.call_log[0]["prompt"]
        self.assertIn(self.q.misconceptions[0].wrong_answer, prompt)
        self.assertNotIn(self.q.misconceptions[0].wrong_answer, prompt.rsplit("Student answer:", 1)[1])

    def test_wording_error_is_wrong_verdict_with_wording_error_type(self):
        s = self._session(make_classifier_fn(MockInferenceClient()))
        r = s.submit_answer("I got the wording mixed up but the tissue regrows itself")
        self.assertEqual((r.kind, r.error_type, r.error_source), ("hint", "wording_error", "llm"))
        answer = self._events(s, "answer")[-1]
        self.assertEqual((answer["verdict"], answer["kind"], answer["error_type"]), ("wrong", "socratic_wrong", "wording_error"))
        self.assertEqual(self.hints, ["wording_error"])
        classified = self._events(s, "llm_classify")[-1]
        self.assertEqual((classified["llm_label"], classified["verdict"]), ("wording_error", "wrong"))

    def test_unknown_label_fails_closed_and_logs_raw(self):
        s = self._session(lambda q, a: {"error_type": "banana", "confidence": 0.99, "reasoning": "?"})
        r = s.submit_answer(self.NATURAL)
        self.assertEqual((r.kind, r.error_type), ("hint", "logic_error"))
        self.assertEqual(self._events(s, "answer")[-1]["verdict"], "wrong")
        classified = self._events(s, "llm_classify")[-1]
        self.assertEqual((classified["llm_label"], classified["error_type"], classified["verdict"]), ("banana", "logic_error", "wrong"))
        self.assertIn("banana", classified["raw_output"])

    def test_classifier_raw_output_reaches_the_log(self):
        # "partial" is a valid label since Phase 3, so an out-of-set one is used here.
        text = "LABEL: halfway\nCONFIDENCE: 0.9\nREASONING: half"
        s = self._session(make_classifier_fn(_ScriptedClient(text)))
        r = s.submit_answer(self.NATURAL)
        self.assertEqual(r.error_type, "logic_error")
        classified = self._events(s, "llm_classify")[-1]
        self.assertEqual(classified["raw_output"], text)
        self.assertEqual(classified["verdict"], "wrong")

    def test_label_partial_line_is_a_partial_verdict(self):
        s = self._session(make_classifier_fn(_ScriptedClient("LABEL: partial\nCONFIDENCE: 0.9\nREASONING: half")))
        r = s.submit_answer(self.NATURAL)
        self.assertEqual((r.kind, r.error_type, r.error_source), ("hint", "logic_error", "llm"))
        answer = self._events(s, "answer")[-1]
        self.assertEqual((answer["verdict"], answer["kind"]), ("partial", "socratic_partial"))
        self.assertEqual(self.hints, ["logic_error"])

    def test_noise_prefix_does_not_make_answers_correct(self):
        # The prefix ends in "present and correct." and rides inside every answer.
        s = self._session(make_classifier_fn(MockInferenceClient()))
        r = s.submit_answer("The damaged tissue just swells up bigger over time")
        self.assertEqual((r.kind, r.error_type), ("hint", "logic_error"))
        self.assertEqual(self._events(s, "answer")[-1]["verdict"], "wrong")

    def test_llm_correct_on_last_attempt_scores_correct_not_escalated(self):
        replies = iter(["logic_error", "logic_error", "correct"])
        s = self._session(lambda q, a: {"error_type": next(replies), "confidence": 0.9, "reasoning": "t"})
        kinds = [s.submit_answer(self.NATURAL).kind for _ in range(3)]
        self.assertEqual(kinds, ["hint", "hint", "correct"])
        self.assertEqual(s.state.stuck_question_ids, [])
        self.assertAlmostEqual(
            s.state.mastery["cell_theory"],
            max(MASTERY_FLOOR, INITIAL_MASTERY + 2 * SOCRATIC_WRONG_DELTA) + SOCRATIC_CORRECT_DELTA,
        )

    def test_key_terms_case_a_turn_result_unchanged(self):
        s = self._session(Mock(side_effect=AssertionError("LLM reached on Case A")))
        s.verify_fn = make_verify_fn(MockInferenceClient())
        r = s.submit_answer("pre-existing cells undergo division to heal the wound")
        self.assertEqual((r.kind, r.error_type, r.error_source), ("correct", None, None))
        answer = self._events(s, "answer")[-1]
        self.assertEqual(
            list(answer),
            ["ts", "event", "question_id", "tier", "answer", "attempt", "correct", "verdict", "kind",
             "error_source", "key_terms_matched", "key_terms_missing", "disengagement_flag", "disengagement_tokens",
             "case_a_verified"],
        )
        self.assertEqual(answer["error_source"], "key_terms_verified")
        self.assertIs(answer["case_a_verified"], True)


if __name__ == "__main__":
    unittest.main()
