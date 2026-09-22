"""Offline tests for hint_pipeline against the mock client.

Run from the project root:  python -m unittest discover -s tests -v
No model, no NPU, no network.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from socratic_core.hint_pipeline import HINT_STRATEGY, hint_pipeline, validate_hint  # noqa: E402
from socratic_core.mock_client import MockInferenceClient  # noqa: E402
from socratic_core.question_bank import Question  # noqa: E402

# Synthetic fixture (not from the live bank): the hints below are written
# around a "the cell" / "atom" vocabulary that no real bank entry uses.
Q1 = Question(
    id="s_test_cell_theory_L1",
    topic="Cell Theory",
    question_text="According to cell theory, what is the basic structural and functional unit of all living organisms?",
    correct_answer="the cell",
    accepted_variants=("cell", "cells"),
    fallback_hint="What is the smallest part of your body that is still alive on its own?",
    min_words=1,
    answer_explanation="Every living thing is made of one or more cells.",
    tier="socratic",
    cluster="cell_theory",
    key_terms=("cell", "alive"),
    level=1,
)

VALID_HINT = "Is an atom alive on its own?"
LEAKING_HINT = "Is it the cell?"
NO_QMARK_HINT = "Think about what is alive."


class _ScriptedClient(MockInferenceClient):
    """Returns the given texts in order; the base mock cannot vary output for
    identical prompts, which the regenerate path requires."""

    def __init__(self, texts: list[str]) -> None:
        super().__init__()
        self._texts = list(texts)

    def _canned_text(self, prompt: str) -> str:
        return self._texts.pop(0)


class ValidatorTests(unittest.TestCase):
    def test_valid_hint_passes_validation(self):
        self.assertEqual(validate_hint(VALID_HINT, Q1), (True, "ok"))

    def test_hint_with_answer_rejected(self):
        ok, reason = validate_hint(LEAKING_HINT, Q1)
        self.assertFalse(ok)
        self.assertIn("correct answer", reason)
        ok, reason = validate_hint("Could it be a CELL?", Q1)
        self.assertFalse(ok)

    def test_hint_too_long_rejected(self):
        hint = " ".join(["word"] * 25) + " why?"
        self.assertEqual(len(hint.split()), 26)
        ok, reason = validate_hint(hint, Q1)
        self.assertFalse(ok)
        self.assertIn("too long", reason)

    def test_hint_without_question_mark_rejected(self):
        ok, reason = validate_hint(NO_QMARK_HINT, Q1)
        self.assertFalse(ok)
        self.assertIn("?", reason)


class PipelineTests(unittest.TestCase):
    def test_pipeline_uses_llm_when_valid(self):
        mock = _ScriptedClient([VALID_HINT])
        r = hint_pipeline(Q1, "the atom", "logic_error", mock)
        self.assertEqual(r, {"hint": VALID_HINT, "source": "llm", "rejections": []})
        self.assertEqual(len(mock.call_log), 1)

    def test_pipeline_regenerates_once_on_invalid(self):
        mock = _ScriptedClient([LEAKING_HINT, VALID_HINT])
        r = hint_pipeline(Q1, "the atom", "logic_error", mock)
        self.assertEqual(r["source"], "llm")
        self.assertEqual(r["hint"], VALID_HINT)
        self.assertEqual(len(r["rejections"]), 1)
        self.assertEqual(len(mock.call_log), 2)

    def test_pipeline_falls_back_to_template_after_two_failures(self):
        mock = _ScriptedClient([LEAKING_HINT, NO_QMARK_HINT])
        r = hint_pipeline(Q1, "the atom", "logic_error", mock)
        self.assertEqual(r["source"], "template")
        self.assertEqual(r["hint"], Q1.fallback_hint)
        self.assertEqual(len(r["rejections"]), 2)
        self.assertEqual(len(mock.call_log), 2)

    def test_pipeline_handles_client_error(self):
        mock = MockInferenceClient(fail_mode=True)
        r = hint_pipeline(Q1, "the atom", "logic_error", mock)
        self.assertEqual(r["source"], "template")
        self.assertEqual(r["hint"], Q1.fallback_hint)
        self.assertEqual(r["rejections"], [])

    def test_generator_does_not_include_correct_answer_in_prompt(self):
        mock = MockInferenceClient()
        hint_pipeline(Q1, "the atom", "logic_error", mock)
        prompt = mock.call_log[0]["prompt"].lower()
        self.assertNotIn(Q1.correct_answer.lower(), prompt)
        self.assertNotIn(Q1.answer_explanation.lower(), prompt)
        self.assertIn(Q1.question_text.lower(), prompt)
        self.assertIn("the atom", prompt)
        self.assertIn("logic_error", prompt)


class PartialStrategyTests(unittest.TestCase):
    """Case B (key_terms partial) must aim the hint at the omitted idea."""

    MISSING = ["alive"]

    def test_partial_hint_mentions_missing_key_term(self):
        # The base mock answers every hint prompt with one canned question that
        # names nothing, so the pipeline falls through to the partial template.
        mock = MockInferenceClient()
        r = hint_pipeline(Q1, "the atom", "logic_error", mock, missing_terms=self.MISSING)
        self.assertEqual(r["source"], "partial_template")
        self.assertIn("alive", r["hint"].lower())
        self.assertNotEqual(r["hint"], Q1.fallback_hint)
        self.assertTrue(validate_hint(r["hint"], Q1)[0])
        self.assertEqual(r["rejections"], ["names no missing key term"] * 2)

        # A model hint that does name the term is kept as-is.
        scripted = _ScriptedClient(["What keeps a thing alive on its own?"])
        r = hint_pipeline(Q1, "the atom", "logic_error", scripted, missing_terms=self.MISSING)
        self.assertEqual(r["source"], "llm")
        self.assertIn("alive", r["hint"].lower())

    def test_partial_hint_differs_from_logic_error_hint(self):
        logic = hint_pipeline(Q1, "the atom", "logic_error", MockInferenceClient())
        partial = hint_pipeline(
            Q1, "the atom", "logic_error", MockInferenceClient(), missing_terms=self.MISSING
        )
        self.assertNotEqual(partial["hint"], logic["hint"])
        self.assertNotEqual(partial["source"], logic["source"])
        self.assertIn("alive", partial["hint"].lower())
        self.assertNotIn("alive", logic["hint"].lower())

    def test_partial_strategy_reaches_the_prompt(self):
        mock = MockInferenceClient()
        hint_pipeline(Q1, "the atom", "logic_error", mock, missing_terms=self.MISSING)
        prompt = mock.call_log[0]["prompt"]
        self.assertIn(HINT_STRATEGY["partial"], prompt)
        self.assertNotIn(HINT_STRATEGY["logic_error"], prompt)
        self.assertIn("alive", prompt)
        # The load-bearing invariant still holds on the new path.
        self.assertNotIn(Q1.correct_answer.lower(), prompt.lower())
        self.assertNotIn(Q1.answer_explanation.lower(), prompt.lower())

    def test_empty_missing_terms_behaves_like_before(self):
        for terms in (None, [], ["", "  "]):
            r = hint_pipeline(Q1, "the atom", "logic_error", MockInferenceClient(), missing_terms=terms)
            with self.subTest(terms=terms):
                self.assertEqual(r["source"], "llm")
                self.assertEqual(r["rejections"], [])


if __name__ == "__main__":
    unittest.main()
