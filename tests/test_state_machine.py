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
from socratic_core.disengagement import DISENGAGEMENT_TOKENS  # noqa: E402
from socratic_core.mastery import (  # noqa: E402
    FILTER_CORRECT_DELTA,
    FILTER_WRONG_DELTA,
    INITIAL_MASTERY,
    MASTERY_CEILING,
    SOCRATIC_CORRECT_DELTA,
    SOCRATIC_PARTIAL_DELTA,
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
        # S_CELL key_terms = ["cell", "division"]: behavioural -> key_terms (A/B/C) -> LLM.
        classified: list[str] = []
        hinted: list[str] = []

        def fake_llm(question, answer):
            classified.append(answer)
            return {"error_type": "wording_error", "confidence": 0.9, "reasoning": "test"}

        def fake_hint(question, answer, error_type):
            hinted.append(error_type)
            return "Is an atom alive on its own?"

        s = SocraticSession(
            self.full, question_ids=["s_001"], classifier_fn=fake_llm, hint_fn=fake_hint, sessions_dir=self.sessions_dir
        )
        r = s.submit_answer("idk")  # behavioural layer
        self.assertEqual((r.kind, r.error_type, r.error_source), ("hint", "low_effort", "behavioural"))
        r = s.submit_answer("cells are the unit of life")  # key_terms: "cell" hit, "division" missing -> partial
        self.assertEqual((r.error_type, r.error_source), ("logic_error", "key_terms"))
        self.assertEqual(classified, [])  # neither turn so far reached the LLM
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
        self.assertEqual(len(classified), 1)
        self.assertTrue(classified[0].startswith(NOISE_TOLERANCE_PREFIX))
        self.assertTrue(classified[0].endswith("the organ system in the body"))
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
        s = SocraticSession(self.full, question_ids=["s_001"], classifier_fn=self.llm, hint_fn=self.hint, sessions_dir=self.sessions_dir)
        r = s.submit_answer("cells form through division of other cells")
        self.assertEqual(r.kind, "correct")
        answer_entry = [h for h in s.state.history if h["event"] == "answer"][0]
        self.assertEqual(answer_entry["kind"], "socratic_correct")
        self.assertEqual(answer_entry["verdict"], "correct")
        self.assertEqual(answer_entry["error_source"], "key_terms")
        self.assertEqual(set(answer_entry["key_terms_matched"]), {"cell", "division"})
        self.assertEqual(answer_entry["key_terms_missing"], [])
        self.llm.assert_not_called()

    def test_socratic_partial_on_partial_key_terms(self):
        s = SocraticSession(self.full, question_ids=["s_001"], classifier_fn=self.llm, hint_fn=lambda q, a, e: "hint?", sessions_dir=self.sessions_dir)
        r = s.submit_answer("cells are the smallest unit")  # "cell" hit, "division" missing
        self.assertEqual(r.kind, "hint")
        self.assertEqual(r.error_type, "logic_error")
        self.assertEqual(r.error_source, "key_terms")

    def test_socratic_partial_records_missing_key_terms(self):
        s = SocraticSession(self.full, question_ids=["s_001"], classifier_fn=self.llm, hint_fn=lambda q, a, e: "hint?", sessions_dir=self.sessions_dir)
        s.submit_answer("cells are the smallest unit")
        answer_entry = [h for h in s.state.history if h["event"] == "answer"][0]
        self.assertEqual(answer_entry["verdict"], "partial")
        self.assertEqual(answer_entry["kind"], "socratic_partial")
        self.assertEqual(answer_entry["key_terms_matched"], ["cell"])
        self.assertEqual(answer_entry["key_terms_missing"], ["division"])

    def test_socratic_partial_skips_llm(self):
        s = SocraticSession(self.full, question_ids=["s_001"], classifier_fn=self.llm, hint_fn=lambda q, a, e: "hint?", sessions_dir=self.sessions_dir)
        s.submit_answer("cells are the smallest unit")
        self.llm.assert_not_called()

    def test_socratic_partial_mastery_delta(self):
        s = SocraticSession(self.full, question_ids=["s_001"], classifier_fn=self.llm, hint_fn=lambda q, a, e: "hint?", sessions_dir=self.sessions_dir)
        s.submit_answer("cells are the smallest unit")
        self.assertAlmostEqual(s.state.mastery["cell_theory"], INITIAL_MASTERY + SOCRATIC_PARTIAL_DELTA)

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

    def test_disengagement_flag_set_on_meme_tokens(self):
        s = SocraticSession(self.full, question_ids=["s_001"], classifier_fn=self.llm, hint_fn=lambda q, a, e: "hint?", sessions_dir=self.sessions_dir)
        s.submit_answer("cells divide skibidi rizz")
        answer_entry = [h for h in s.state.history if h["event"] == "answer"][0]
        self.assertTrue(answer_entry["disengagement_flag"])
        self.assertEqual(set(answer_entry["disengagement_tokens"]), {"skibidi", "rizz"})

    def test_disengagement_flag_not_set_on_clean_answer(self):
        s = SocraticSession(self.full, question_ids=["s_001"], classifier_fn=self.llm, hint_fn=lambda q, a, e: "hint?", sessions_dir=self.sessions_dir)
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
        # Property under test: disengagement flag fires but does NOT alter
        # the verdict produced by the pipeline.
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


if __name__ == "__main__":
    unittest.main()
