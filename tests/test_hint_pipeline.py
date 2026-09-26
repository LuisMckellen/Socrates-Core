"""Offline tests for hint_pipeline against the mock client.

Run from the project root:  python -m unittest discover -s tests -v
No model, no NPU, no network.
"""

from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from socratic_core.hint_pipeline import (  # noqa: E402
    HINT_STRATEGY,
    KEY_TERM_FORMS,
    PARAPHRASE_NGRAM,
    SYSTEM_PROMPT,
    build_hint_prompt,
    hint_pipeline,
    key_term_forms,
    named_key_terms,
    shared_answer_ngrams,
    validate_hint,
)
from socratic_core.inference_client import build_prompt  # noqa: E402
from socratic_core.mock_client import MockInferenceClient  # noqa: E402
from socratic_core.question_bank import Misconception, Question, load_question_bank  # noqa: E402

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
        # Counted, so a client-caused fallback shows up in the answer event's rejections.
        self.assertEqual(r["rejections"], ["client error"])

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
    """A partial verdict (from the LLM classifier) must aim the hint at the omitted idea."""

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


# Q1 with bank misconceptions and a natural_correct_example, for the targeted path.
QT = dataclasses.replace(
    Q1,
    natural_correct_example="Your body is built from tiny living building blocks",
    misconceptions=(
        Misconception("ct_atom_is_unit", "The atom is the smallest unit of life", "logic_error", "Atoms are not alive."),
        Misconception(
            "ct_organ_is_unit", "The organ is the unit", "logic_error", "Organs are made of cells.",
            diagnostic_goal="Notice that an organ is made of smaller living parts.",
        ),
    ),
)

# Live bank: rule 3b is tuned against this question's real Local CPU output.
LIVE = load_question_bank(ROOT / "question_bank.json").get("s_cell_theory_L1")
# Every distinct hint the model produced on s_cell_theory_L1 during the step 0a traces.
REAL_HINTS = (
    "Can a single swollen cell replace an entire tissue with complex structure and function?",
    "If a cell doesn't divide, can tissue grow or wounds heal? What would happen if pre-existing cells failed to divide?",
    "If a cell doesn't divide, can tissue grow or wounds heal? Why or why not?",
    "If a tissue grows without any pre-existing cells, where do the new cells come from?",
    "If a tissue heals by simply swelling, what happens to the number of cells in the repaired area?",
    "If a tissue simply swells without new cells, what happens to its structure after healing?",
    "If a tissue simply swells without new cells, what happens to its structure over time?",
)
# Paraphrase leaks that clear the verbatim rules.
PARAPHRASE_PROBES = (
    "Do cells already in the body divide to fill the damaged area?",  # natural_correct_example, reworded
    "Could pre-existing cells undergo division to form replacement cells?",  # correct_answer, reworded
)


class TargetedPromptTests(unittest.TestCase):
    ANSWER = "the atom I think"

    def _prompt(self, question=QT, **kw):
        return build_hint_prompt(question, self.ANSWER, "logic_error", **kw)

    def _untargeted(self, question=QT):
        # The pre-0c construction, restated: the untargeted prompt must not drift.
        user = "\n".join([
            f"Question: {question.question_text}",
            f"Student answer: {self.ANSWER}",
            "Error type: logic_error",
            HINT_STRATEGY["logic_error"],
        ])
        return build_prompt(user, system=SYSTEM_PROMPT)

    def test_targeted_prompt_fields(self):
        prompt = self._prompt(matched_bank_id="ct_atom_is_unit")
        for expected in (
            f"Topic: {QT.topic}\n",
            f"Question: {QT.question_text}\n",
            f"Student answer: {self.ANSWER}\n",  # the student's verbatim answer is the belief to extend
            f"paraphrase or confirm it): {QT.natural_correct_example}\n",
            HINT_STRATEGY["misconception"],
        ):
            self.assertIn(expected, prompt)
        self.assertTrue(HINT_STRATEGY["misconception"].startswith("The student's answer states a mistaken belief."))
        self.assertNotIn("Error type:", prompt)
        self.assertNotIn("Diagnostic goal", prompt)  # ct_atom_is_unit has none
        self.assertNotIn("None", prompt)
        # Still a different prompt from the untargeted one.
        self.assertNotEqual(prompt, self._untargeted())

    def test_targeted_prompt_never_carries_wrong_answer(self):
        # 0c ablation: with wrong_answer in the prompt the model put the bank's
        # wording ("swell") into hints for students who never used it.
        for question, mid in ((QT, "ct_atom_is_unit"), (QT, "ct_organ_is_unit"),
                              (LIVE, "ct_hypertrophy"), (LIVE, "ct_spontaneous_generation")):
            misconception = next(m for m in question.misconceptions if m.id == mid)
            with self.subTest(matched_bank_id=mid):
                prompt = build_hint_prompt(question, "cells just get bigger", "logic_error", matched_bank_id=mid)
                self.assertIn("Student answer: cells just get bigger\n", prompt)
                self.assertNotIn(misconception.wrong_answer, prompt)
                self.assertNotIn(misconception.wrong_answer.lower(), prompt.lower())
                self.assertNotIn("common mistake", prompt)

    def test_targeted_prompt_never_carries_bank_answer_fields(self):
        # On the live entry (QT's "the cell" also appears in its own question text).
        prompt = build_hint_prompt(LIVE, "cells swell up", "logic_error", matched_bank_id="ct_hypertrophy")
        self.assertIn(LIVE.natural_correct_example, prompt)
        for secret in (LIVE.correct_answer, LIVE.answer_explanation, *LIVE.accepted_variants):
            with self.subTest(secret=secret):
                self.assertNotIn(secret, prompt)

    def test_diagnostic_goal_line_only_when_present(self):
        with_goal = self._prompt(matched_bank_id="ct_organ_is_unit")
        self.assertIn("Diagnostic goal: Notice that an organ is made of smaller living parts.\n", with_goal)
        for goal in (None, "", "   "):
            blank = dataclasses.replace(QT, misconceptions=(dataclasses.replace(QT.misconceptions[1], diagnostic_goal=goal),))
            with self.subTest(goal=goal):
                prompt = self._prompt(blank, matched_bank_id="ct_organ_is_unit")
                self.assertNotIn("Diagnostic goal", prompt)
                self.assertNotIn("None", prompt)
        # No natural_correct_example: that line is omitted too.
        no_natural = self._prompt(dataclasses.replace(QT, natural_correct_example=""), matched_bank_id="ct_atom_is_unit")
        self.assertNotIn("Correct idea", no_natural)

    def test_unmatched_ids_keep_untargeted_prompt(self):
        # Not one of this question's misconceptions -> byte-identical to the pre-0c prompt.
        for mid in (None, "none", "natural_correct", "partial_example", "enz_shifts_equilibrium", "m_0"):
            with self.subTest(matched_bank_id=mid):
                self.assertEqual(self._prompt(matched_bank_id=mid), self._untargeted())
        self.assertEqual(self._prompt(), self._untargeted())

    def test_partial_wins_over_matched_misconception(self):
        prompt = self._prompt(matched_bank_id="ct_atom_is_unit", missing_terms=["alive"])
        self.assertIn(HINT_STRATEGY["partial"], prompt)
        self.assertIn("Ideas the student left out: alive", prompt)
        self.assertNotIn(HINT_STRATEGY["misconception"], prompt)
        self.assertNotIn(QT.natural_correct_example, prompt)

    def test_pipeline_targets_matched_misconception(self):
        # Names no key term (QT's are "cell" and "alive"), so rule 3c lets it through.
        hint = "What would an atom need to do on its own?"
        mock = _ScriptedClient([hint])
        r = hint_pipeline(QT, self.ANSWER, "logic_error", mock, matched_bank_id="ct_atom_is_unit")
        self.assertEqual(r, {"hint": hint, "source": "llm", "rejections": []})
        self.assertIn(HINT_STRATEGY["misconception"], mock.call_log[0]["prompt"])

    def test_rule_3c_only_on_targeted_path(self):
        # VALID_HINT names "alive": fine untargeted (as before 0c), rejected when targeted.
        targeted = hint_pipeline(QT, self.ANSWER, "logic_error", _ScriptedClient([VALID_HINT, VALID_HINT]),
                                 matched_bank_id="ct_atom_is_unit")
        self.assertEqual(targeted["source"], "template")
        self.assertEqual(targeted["rejections"], ["names a key term on the targeted path"] * 2)
        for mid in (None, "none", "partial_example"):
            with self.subTest(matched_bank_id=mid):
                r = hint_pipeline(QT, self.ANSWER, "logic_error", _ScriptedClient([VALID_HINT]), matched_bank_id=mid)
                self.assertEqual(r, {"hint": VALID_HINT, "source": "llm", "rejections": []})

    def test_pipeline_regenerates_then_falls_back_on_targeted_leak(self):
        leaks = ["Is your body built from tiny living building blocks?", "Your body is built from tiny living building blocks, right?"]
        mock = _ScriptedClient(leaks)
        r = hint_pipeline(QT, self.ANSWER, "logic_error", mock, matched_bank_id="ct_atom_is_unit")
        self.assertEqual(r["source"], "template")
        self.assertEqual(r["hint"], QT.fallback_hint)
        self.assertEqual(len(r["rejections"]), 2)
        self.assertEqual(len(mock.call_log), 2)


class AnswerLeakRuleTests(unittest.TestCase):
    def test_rule_3a_rejects_natural_correct_example(self):
        for hint in (
            "So, your body is built from tiny living building blocks?",
            "YOUR BODY is built from tiny living building-blocks -- right?",
        ):
            with self.subTest(hint=hint):
                self.assertEqual(validate_hint(hint, QT), (False, "contains natural_correct_example"))
        self.assertEqual(validate_hint(VALID_HINT, QT), (True, "ok"))
        # Without the field the rule is inert (and never matches an empty string).
        self.assertTrue(validate_hint(VALID_HINT, dataclasses.replace(QT, natural_correct_example=""))[0])

    def test_rule_3b_catches_paraphrase_probes(self):
        for hint in PARAPHRASE_PROBES:
            with self.subTest(hint=hint):
                ok, reason = validate_hint(hint, LIVE)
                self.assertFalse(ok)
                self.assertTrue(reason.startswith("paraphrases the answer: "), reason)

    def test_rule_3b_accepts_real_hints(self):
        for hint in REAL_HINTS:
            with self.subTest(hint=hint):
                self.assertEqual(validate_hint(hint, LIVE), (True, "ok"))

    def test_rule_3b_window_is_four(self):
        # 5-word probe: a 5-word window still accepts every real hint but misses the
        # first paraphrase probe, so 4 is the widest window that catches both.
        self.assertEqual(PARAPHRASE_NGRAM, 4)
        self.assertEqual([h for h in REAL_HINTS if shared_answer_ngrams(h, LIVE, n=5)], [])
        self.assertFalse(shared_answer_ngrams(PARAPHRASE_PROBES[0], LIVE, n=5))
        self.assertTrue(shared_answer_ngrams(PARAPHRASE_PROBES[1], LIVE, n=5))
        self.assertTrue(all(shared_answer_ngrams(p, LIVE, n=4) for p in PARAPHRASE_PROBES))

    def test_rule_3b_does_not_catch_synonym_rewrite(self):
        # Documented limit: no shared run of content words, so nothing to match.
        self.assertTrue(validate_hint("What if existing cells split to fill in the hurt area?", LIVE)[0])

    def test_rule_3c_rejects_either_or_hints_from_0c_traces(self):
        # The six targeted hints Local CPU produced under the first 0c strategy line:
        # four offer "divide" (a KEY_TERM_FORMS form of "division"), two say
        # "existing" (a synonyms.py entry for "pre-existing").
        for hint in (
            "Do cells in a wound grow larger to fill space, or do they divide to produce more cells?",
            "Do cells in a wound grow larger to fill the space, or do they divide to produce more cells?",
            "Do new cells form in a wound to replace lost ones, or do existing cells just grow larger?",
            "Do cells at a wound site divide to produce more cells, or do new ones appear from nothing?",
            "Do you think new cells appear without any existing cells being present to divide?",
            "Do cells at a wound site divide to produce more cells, or do they appear from nothing?",
        ):
            with self.subTest(hint=hint):
                self.assertEqual(validate_hint(hint, LIVE, targeted=True), (False, "names a key term on the targeted path"))
                self.assertTrue(validate_hint(hint, LIVE)[0])  # untargeted: unchanged

    def test_rule_3c_matches_term_synonym_form_and_stem(self):
        for hint, hit in (
            ("What does division change?", "division~division"),        # the term itself
            ("Could they be preexisting?", "pre-existing~preexisting"),  # synonyms.py
            ("Why would a cell divide?", "division~divide"),             # KEY_TERM_FORMS
            ("What happens when it divides?", "division~divides"),       # form, via crude_stem too
            ("Is anything pre-existing here?", "pre-existing~pre-existing"),
        ):
            with self.subTest(hint=hint):
                self.assertIn(hit, named_key_terms(hint, LIVE))
        self.assertEqual(named_key_terms("Where does the extra tissue come from?", LIVE), [])
        self.assertEqual(key_term_forms("division"), ["division", "divide", "divides", "dividing", "divided"])

    def test_key_term_forms_keys_are_live_key_terms(self):
        # Guards 0b's additions against typos: every entry must name a real key term.
        live_terms = {t.lower() for q in load_question_bank(ROOT / "question_bank.json") for t in q.key_terms}
        self.assertEqual(sorted(set(KEY_TERM_FORMS) - live_terms), [])
        self.assertTrue(all(k == k.lower() for k in KEY_TERM_FORMS))


class QuestionMarkTests(unittest.TestCase):
    def test_double_question_mark_rejected(self):
        self.assertEqual(validate_hint("Is an atom alive on its own??", Q1), (False, "ends with '??'"))
        self.assertEqual(validate_hint("Is an atom alive on its own???", Q1), (False, "ends with '??'"))
        self.assertEqual(validate_hint(VALID_HINT, Q1), (True, "ok"))


if __name__ == "__main__":
    unittest.main()
