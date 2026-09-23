"""Offline tests for question_bank, classifier_behavioral and state_machine.

Run from the project root:  python -m unittest discover -s tests -v
No model, no NPU, no network.

Q1/Q2 are synthetic Socratic fixtures (direct ``Question``/``Misconception``
construction, bypassing bank-JSON validation) rather than ids from the live
16-entry ``question_bank.json`` -- the real bank's filter/socratic split and
mastery-escalation flow don't match what these tests exercise (a simple
two-question walk with no filters). See test_state_machine.py for tests
against the live bank's tier/escalation behaviour.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from socratic_core import classifier_behavioral as beh  # noqa: E402
from socratic_core.classifier_behavioral import classify_behavioural  # noqa: E402
from socratic_core.disengagement import flag_disengagement  # noqa: E402
from socratic_core.question_bank import (  # noqa: E402
    Misconception,
    Question,
    QuestionBank,
    QuestionBankError,
    load_question_bank,
    normalize_answer,
)
from socratic_core.state_machine import SocraticSession  # noqa: E402

Q1 = Question(
    id="s_test_cell_theory_L1",
    topic="Cell Theory",
    question_text="According to cell theory, what is the basic structural and functional unit of all living organisms?",
    correct_answer="the cell",
    accepted_variants=("cell", "cells"),
    misconceptions=(
        Misconception("the atom", "logic_error", "Atoms are not alive; the basic unit of life is the cell."),
        Misconception("the organ is the unit", "logic_error", "Organs are made of many cells working together."),
    ),
    fallback_hint="What is the smallest part of your body that is still alive on its own?",
    min_words=1,
    answer_explanation="Every living thing is made of one or more cells.",
    tier="socratic",
    cluster="cell_theory",
    key_terms=("cell", "alive"),
    level=1,
)
Q2 = Question(
    id="s_test_cell_theory_L2",
    topic="Cell Theory",
    question_text="According to cell theory, where do all new cells come from?",
    correct_answer="from pre-existing cells",
    accepted_variants=("pre-existing cells", "from other cells", "existing cells"),
    misconceptions=(
        Misconception("they form on their own", "logic_error", "This is spontaneous generation, which experiments disproved."),
        Misconception("cells make cells", "logic_error", "Right idea, but state it precisely: new cells arise from pre-existing cells."),
    ),
    fallback_hint="What must already exist for a new cell to form?",
    min_words=2,
    answer_explanation="New cells arise only from the division of pre-existing cells.",
    tier="socratic",
    cluster="cell_theory",
    key_terms=("cell", "divide"),
    level=2,
)
BANK = QuestionBank([Q1, Q2])


class QuestionBankTests(unittest.TestCase):
    def test_loads_all_questions(self):
        self.assertGreaterEqual(len(BANK), 1)
        required = (
            "id",
            "topic",
            "question_text",
            "correct_answer",
            "accepted_variants",
            "min_words",
            "misconceptions",
            "fallback_hint",
        )
        for q in BANK:
            with self.subTest(question_id=q.id):
                for name in required:
                    # Question is a dataclass with defaults, so presence alone
                    # proves nothing: assert the bank actually populated it.
                    self.assertTrue(getattr(q, name), f"empty or missing: {name}")
                self.assertIsInstance(q.min_words, int)
                self.assertGreaterEqual(q.min_words, 1)
                self.assertTrue(q.fallback_hint.strip().endswith("?"))
                for m in q.misconceptions:
                    with self.subTest(wrong_answer=m.wrong_answer):
                        self.assertTrue(m.wrong_answer)
                        self.assertIn(m.error_type, ("wording_error", "logic_error"))
                        self.assertTrue(m.explanation)

    def test_normalize_answer_strips_filler(self):
        self.assertEqual(normalize_answer("I think it's the Cell."), "cell")
        self.assertEqual(normalize_answer("The answer is a cell"), "cell")

    def test_is_correct_exact_and_variants(self):
        for a in ("the cell", "Cell", "cells", "it's the cell", "I think cells"):
            self.assertTrue(Q1.is_correct(a), a)
        self.assertTrue(Q2.is_correct("Pre-existing cells"))
        self.assertTrue(Q2.is_correct("from other cells"))

    def test_is_correct_rejects_substrings(self):
        self.assertFalse(Q1.is_correct("cell membrane"))
        self.assertFalse(Q1.is_correct(""))
        self.assertFalse(Q1.is_correct("atom"))

    def test_match_misconception(self):
        m = Q1.match_misconception("The Atom!")
        self.assertIsNotNone(m)
        self.assertEqual(m.error_type, "logic_error")
        self.assertIsNone(Q1.match_misconception("a banana"))

    def test_keywords_include_topic_and_answers(self):
        kw = Q1.keywords()
        self.assertIn("cell", kw)
        self.assertIn("theory", kw)
        self.assertIn("atom", kw)
        self.assertNotIn("the", kw)

    def test_bad_bank_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.json"
            p.write_text(json.dumps({"questions": [{"id": "x"}]}), encoding="utf-8")
            with self.assertRaises(QuestionBankError):
                load_question_bank(p)
            p.write_text(
                json.dumps(
                    {
                        "questions": [
                            {
                                "id": "x",
                                "topic": "t",
                                "question_text": "q",
                                "correct_answer": "a",
                                "fallback_hint": "no question mark",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(QuestionBankError):
                load_question_bank(p)

    def _load_one(self, raw: dict):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bank.json"
            p.write_text(json.dumps({"questions": [raw]}), encoding="utf-8")
            return load_question_bank(p)

    _SOCRATIC_RAW = {
        "id": "s_x", "tier": "socratic", "cluster": "c", "topic": "t", "level": 1,
        "question_text": "q", "correct_answer": "a", "accepted_variants": ["a"],
        "key_terms": ["a"], "min_words": 3, "fallback_hint": "h?",
        "misconceptions": [{"wrong_answer": "w", "error_type": "logic_error", "explanation": "e"}],
    }
    _FILTER_RAW = {
        "id": "f_x", "tier": "filter", "cluster": "c", "topic": "t",
        "question_text": "q", "correct_answer": "a", "accepted_variants": ["a"],
        "min_words": 1, "fallback_hint": "",
    }

    def test_natural_correct_example_optional_on_socratic(self):
        self.assertEqual(self._load_one(dict(self._SOCRATIC_RAW)).get("s_x").natural_correct_example, "")
        q = self._load_one({**self._SOCRATIC_RAW, "natural_correct_example": "old ones split"}).get("s_x")
        self.assertEqual(q.natural_correct_example, "old ones split")

    def test_natural_correct_example_must_be_string(self):
        with self.assertRaises(QuestionBankError):
            self._load_one({**self._SOCRATIC_RAW, "natural_correct_example": 3})

    def test_filter_rejects_natural_correct_example(self):
        self.assertEqual(self._load_one(dict(self._FILTER_RAW)).get("f_x").natural_correct_example, "")
        with self.assertRaises(QuestionBankError):
            self._load_one({**self._FILTER_RAW, "natural_correct_example": ""})


class BehavioralClassifierTests(unittest.TestCase):
    def test_idk_with_context_is_low_effort(self):
        # answer contains "idk" but exceeds min_words
        # must return low_effort, never logic_error
        for question in (Q1, Q2):
            result = classify_behavioural("idk I didn't study this", question)
            self.assertEqual(result, "low_effort")
        self.assertIn("give-up phrase", beh.explain("idk I didn't study this", Q1))

    def test_keyword_fires_before_min_words(self):
        # 'skip' alone is 1 word; the recorded reason must be the keyword, not length.
        self.assertIn("give-up phrase", beh.explain("skip", Q2))

    def test_min_words_is_per_question_lower_bound(self):
        self.assertEqual(Q1.min_words, 1)
        self.assertEqual(Q2.min_words, 2)
        self.assertEqual(classify_behavioural("", Q1), "low_effort")
        self.assertEqual(classify_behavioural("atom", Q2), "low_effort")  # 1 < 2
        self.assertIsNone(classify_behavioural("atom", Q1))  # 1 >= 1 -> next layer

    def test_no_upper_bound_on_length(self):
        long_answer = (
            "I believe new cells arise when an existing cell copies its "
            "material and then divides into two daughter cells "
        ) * 5
        self.assertIsNone(classify_behavioural(long_answer, Q2))

    def test_give_up_phrases(self):
        for a in ("idk lol whatever", "I don't know this one", "can we skip this", "no idea honestly"):
            self.assertEqual(classify_behavioural(a, Q1), "low_effort", a)

    def test_off_topic_alone_falls_through(self):
        # Option B: off-topic is a routing signal, not a verdict. Without a
        # disengagement token it must reach key_terms and then the LLM.
        answer = "my favourite football team won yesterday"
        self.assertTrue(beh.is_off_topic(answer, Q1))
        self.assertIsNone(classify_behavioural(answer, Q1))
        self.assertIsNone(beh.explain(answer, Q1))

    def test_off_topic_natural_language_falls_through(self):
        # T1: a genuine paraphrase sharing no stemmed vocabulary with the
        # question. This is the case Option B exists to rescue.
        answer = "The body makes more of itself when you get hurt"
        self.assertTrue(beh.is_off_topic(answer, Q1))
        self.assertFalse(flag_disengagement(answer)[0])
        self.assertIsNone(classify_behavioural(answer, Q1))

    def test_off_topic_with_disengagement_token_is_low_effort(self):
        # T2: off-topic *and* a meme token is the one combination that still
        # produces a verdict at the gate.
        answer = "skibidi my favourite football team won yesterday"
        self.assertTrue(beh.is_off_topic(answer, Q1))
        self.assertTrue(flag_disengagement(answer)[0])
        self.assertEqual(classify_behavioural(answer, Q1), "low_effort")
        self.assertIn("disengagement", beh.explain(answer, Q1))

    def test_no_idea_is_low_effort_via_give_up_phrase(self):
        # T3 regression guard: "No idea" must never fall through. It is caught
        # by rule 1, not by min_words -- Q1.min_words is 1.
        self.assertEqual(classify_behavioural("No idea", Q1), "low_effort")
        self.assertIn("give-up phrase", beh.explain("No idea", Q1))
        self.assertFalse(beh.is_off_topic("No idea", Q1))

    def test_on_topic_wrong_answer_returns_none(self):
        # Genuine attempt: must fall through to the next layer.
        self.assertIsNone(classify_behavioural("the organ system of the body", Q1))
        self.assertIsNone(classify_behavioural("cells come from the nucleus splitting", Q2))

    def test_phrase_is_whole_word(self):
        # 'skip' inside another word must not trigger.
        self.assertIsNone(classify_behavioural("skipping ahead, cells divide from other cells", Q2))


class StateMachineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sessions_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _session(self, **kw):
        return SocraticSession(BANK, sessions_dir=self.sessions_dir, **kw)

    def test_correct_first_try_advances(self):
        s = self._session()
        self.assertEqual(s.current_question().id, Q1.id)
        r = s.submit_answer("the cell")
        self.assertEqual(r.kind, "correct")
        self.assertEqual(r.attempt, 1)
        self.assertEqual(s.current_question().id, Q2.id)
        self.assertEqual(s.state.attempt_count, 0)
        self.assertEqual(s.state.solved_question_ids, [Q1.id])

    def test_three_wrong_escalates_with_two_line_reveal(self):
        s = self._session()
        r1 = s.submit_answer("the atom")  # zero key_terms ("cell", "alive") -> LLM (default)
        self.assertEqual(r1.kind, "hint")
        self.assertEqual(r1.error_type, "logic_error")
        self.assertEqual(r1.error_source, "llm")
        self.assertEqual(r1.message, Q1.fallback_hint)  # default hint_fn

        r2 = s.submit_answer("idk")  # behavioural
        self.assertEqual(r2.kind, "hint")
        self.assertEqual(r2.error_type, "low_effort")
        self.assertEqual(r2.error_source, "behavioural")

        r3 = s.submit_answer("the organ system in the body")  # zero key_terms -> LLM (default)
        self.assertEqual(r3.kind, "escalated")
        self.assertEqual(r3.error_source, "llm")
        self.assertEqual(r3.error_type, "logic_error")
        lines = r3.message.split("\n")
        self.assertEqual(len(lines), 2)
        self.assertIn(Q1.correct_answer, lines[0])
        self.assertEqual(s.state.stuck_question_ids, [Q1.id])
        self.assertEqual(s.current_question().id, Q2.id)
        self.assertEqual(s.state.attempt_count, 0)

    def test_layer_order_keyword_min_words_key_terms_then_llm(self):
        calls: list[str] = []

        def fake_llm(question, answer):
            calls.append(answer)
            return {"error_type": "wording_error", "confidence": 0.9, "reasoning": "test"}

        s = self._session(classifier_fn=fake_llm)
        r = s.submit_answer("idk I didn't study this")  # keyword -> LLM must NOT be called
        self.assertEqual((r.error_type, r.error_source), ("low_effort", "behavioural"))
        self.assertEqual(calls, [])
        r = s.submit_answer("an atom is a cell")  # "cell" hit, "alive" missing: no Case B since Phase 3 -> LLM
        self.assertEqual((r.error_type, r.error_source), ("wording_error", "llm"))
        self.assertEqual(len(calls), 1)
        calls.clear()

        s = self._session(classifier_fn=fake_llm, question_ids=[Q2.id])
        r = s.submit_answer("atom")  # 1 word < min_words=2 -> LLM must NOT be called
        self.assertEqual((r.error_type, r.error_source), ("low_effort", "behavioural"))
        self.assertEqual(calls, [])

        s2 = self._session(classifier_fn=fake_llm)
        r = s2.submit_answer("the organ system in the body")  # zero key_terms -> LLM
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].endswith("the organ system in the body"))
        self.assertEqual(r.error_type, "wording_error")
        self.assertEqual(r.error_source, "llm")

    def test_bypass_bank_lookup_is_now_a_no_op(self):
        # Change 2 removed the bank-misconception lookup layer. "the atom" is a
        # verbatim Q1 misconception (the exact input the old flag governed) and
        # matches no key_terms, so it must reach the LLM regardless of the flag.
        self.assertIsNotNone(Q1.match_misconception("the atom"))

        for flag in (False, True):
            with self.subTest(bypass_bank_lookup=flag):
                calls: list[str] = []

                def fake_llm(question, answer):
                    calls.append(answer)
                    return {"error_type": "wording_error", "confidence": 0.9, "reasoning": "test"}

                s = SocraticSession(
                    BANK, bypass_bank_lookup=flag, classifier_fn=fake_llm, sessions_dir=self.sessions_dir
                )
                r = s.submit_answer("the atom")
                self.assertEqual(len(calls), 1)  # LLM reached; the bank never pre-empted it
                self.assertEqual((r.error_type, r.error_source), ("wording_error", "llm"))
                self.assertNotIn("bank", [h.get("error_source") for h in s.state.history])
                self.assertEqual(s.state.bypass_bank_lookup, flag)

    def test_injected_hint_fn_receives_error_type(self):
        seen = {}

        def fake_hint(question, answer, error_type):
            seen["error_type"] = error_type
            return "Is an atom alive?"

        s = self._session(hint_fn=fake_hint)
        r = s.submit_answer("the atom")
        self.assertEqual(r.message, "Is an atom alive?")
        self.assertEqual(seen["error_type"], "logic_error")

    def test_failing_llm_degrades_gracefully(self):
        def boom(question, answer):
            raise RuntimeError("npu offline")

        s = self._session(classifier_fn=boom, hint_fn=boom)
        r = s.submit_answer("the organ system in the body")
        self.assertEqual(r.kind, "hint")
        self.assertEqual(r.error_type, "logic_error")
        self.assertEqual(r.message, Q1.fallback_hint)
        events = [h["event"] for h in s.state.history]
        self.assertIn("classifier_error", events)
        self.assertIn("hint_error", events)

    def test_session_finishes_and_log_written(self):
        # Scoped to one question so the test does not couple to bank size.
        s = self._session(question_ids=[Q1.id])
        r = s.submit_answer("cell")
        self.assertTrue(r.session_finished)
        self.assertTrue(s.state.finished)
        self.assertIsNone(s.current_question())
        with self.assertRaises(RuntimeError):
            s.submit_answer("anything")

        log = json.loads(s.log_path.read_text(encoding="utf-8"))
        self.assertEqual(log["session_id"], s.state.session_id)
        self.assertEqual(log["solved_question_ids"], [Q1.id])
        self.assertIsNotNone(log["finished_at"])
        events = [h["event"] for h in log["history"]]
        self.assertEqual(events[0], "question_presented")
        self.assertEqual(events[-1], "session_finished")
        self.assertEqual(len(list(self.sessions_dir.glob("*.json"))), 1)

    def test_unknown_question_id_rejected(self):
        with self.assertRaises(KeyError):
            self._session(question_ids=["nope"])


if __name__ == "__main__":
    unittest.main()
