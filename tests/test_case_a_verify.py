"""Offline tests for Case A verification (all key terms present -> YES/NO check).

Run from the project root:  python -m pytest tests/test_case_a_verify.py
No model: the mock answers the verify prompt YES, or NO on a wording/logic steer.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from socratic_core.classifier_llm import (  # noqa: E402
    SYSTEM_PROMPT,
    VERIFY_SYSTEM_PROMPT,
    build_classifier_prompt,
    build_verify_prompt,
    make_classifier_fn,
    make_verify_fn,
    verify_case_a,
)
from socratic_core.hint_pipeline import SYSTEM_PROMPT as HINT_SYSTEM_PROMPT  # noqa: E402
from socratic_core.mock_client import (  # noqa: E402
    CASE_A_VERIFY_MARKER,
    CLASSIFIER_PROMPT_MARKER,
    HINT_PROMPT_MARKER,
    MockInferenceClient,
)
from socratic_core.question_bank import load_question_bank  # noqa: E402
from socratic_core.state_machine import SocraticSession  # noqa: E402

BANK = load_question_bank(ROOT / "question_bank.json")
Q = BANK.get("s_cell_theory_L1")
CLEAN = "pre-existing cells undergo division to form new tissue"
# Hits both key terms ("division", "pre-existing") yet contradicts itself.
CONTRADICTION = "pre-existing cells undergo division because they don't undergo division"
NATURAL = "The body makes more of itself when you get hurt"  # no key terms -> Case C


class _VerifyNoClient(MockInferenceClient):
    """Answers NO to every verify prompt; the classify prompt behaves as the mock."""

    def _canned_text(self, prompt: str) -> str:
        return "NO" if CASE_A_VERIFY_MARKER in prompt else super()._canned_text(prompt)


class _ScriptedClient(MockInferenceClient):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text

    def _canned_text(self, prompt: str) -> str:
        return self.text


class _RaisingClient:
    def generate(self, prompt, max_tokens=256):
        raise RuntimeError("npu fell over")


class CaseAVerifyTests(unittest.TestCase):
    def setUp(self):
        self.sessions_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def _session(self, verify_fn, classifier_client=None):
        return SocraticSession(
            BANK, question_ids=[Q.id],
            classifier_fn=make_classifier_fn(classifier_client or MockInferenceClient()),
            verify_fn=verify_fn, hint_fn=lambda q, a, e: "hint?", sessions_dir=self.sessions_dir,
        )

    @staticmethod
    def _events(s, event):
        return [h for h in s.state.history if h["event"] == event]

    def test_case_a_contradiction_falls_through(self):
        # Guard: the answer must really reach Case A, or this test proves nothing.
        self.assertEqual(SocraticSession._classify_by_key_terms(Q, CONTRADICTION)[0], "correct")
        client = _VerifyNoClient()
        s = self._session(make_verify_fn(client), client)
        r = s.submit_answer(CONTRADICTION)
        self.assertNotEqual(r.kind, "correct")
        answer = self._events(s, "answer")[-1]
        self.assertNotEqual(answer["verdict"], "correct")
        self.assertEqual(answer["error_source"], "llm")
        self.assertIs(answer["case_a_verified"], False)
        self.assertEqual(len(self._events(s, "llm_classify")), 1)  # fell through to Case C

    def test_case_a_clean_still_passes(self):
        s = self._session(make_verify_fn(MockInferenceClient()))
        r = s.submit_answer(CLEAN)
        self.assertEqual(r.kind, "correct")
        answer = self._events(s, "answer")[-1]
        self.assertEqual(
            (answer["verdict"], answer["error_source"], answer["case_a_verified"]),
            ("correct", "key_terms_verified", True),
        )
        self.assertEqual(self._events(s, "llm_classify"), [])  # the classifier never ran

    def test_case_a_verification_error_fails_open(self):
        # A client that raises, or reports an error, inside verify_case_a.
        for client in (_RaisingClient(), MockInferenceClient(fail_mode=True)):
            with self.subTest(client=type(client).__name__):
                self.assertTrue(verify_case_a(Q, CLEAN, client))
        # A verify_fn that raises inside the state machine.
        def boom(question, answer):
            raise RuntimeError("verifier crashed")

        s = self._session(boom)
        r = s.submit_answer(CONTRADICTION)
        self.assertEqual(r.kind, "correct")
        answer = self._events(s, "answer")[-1]
        self.assertEqual((answer["error_source"], answer["case_a_verified"]), ("key_terms_verified", True))
        error = self._events(s, "classifier_error")[-1]
        self.assertEqual(error["stage"], "case_a_verify")

    def test_verify_client_error_logs_classifier_error(self):
        class _ErrorClient:
            def generate(self, prompt, max_tokens=256):
                return {"text": "", "ttft_ms": 0.0, "total_ms": 0.0, "tokens_generated": 0,
                        "error": "model not loaded"}

        s = self._session(make_verify_fn(_ErrorClient()))
        r = s.submit_answer(CONTRADICTION)
        # Still fails open: the verdict is unchanged...
        self.assertEqual(r.kind, "correct")
        answer = self._events(s, "answer")[-1]
        self.assertEqual((answer["error_source"], answer["case_a_verified"]), ("key_terms_verified", True))
        # ...but no longer silently.
        errors = [e for e in self._events(s, "classifier_error") if e.get("stage") == "case_a_verify"]
        self.assertEqual(len(errors), 1)
        self.assertIn("model not loaded", errors[0]["error"])
        # The pure function's contract is untouched: a client error is just True.
        self.assertIs(verify_case_a(Q, CONTRADICTION, _ErrorClient()), True)

    def test_verify_reply_parsing(self):
        for text, expected in (("YES", True), ("no.", False), (" No, it contradicts itself", False),
                               ("Yes", True), ("maybe", True), ("", True)):
            with self.subTest(text=text):
                self.assertIs(verify_case_a(Q, CLEAN, _ScriptedClient(text)), expected)

    def test_mock_dispatches_case_a_verify_prompt(self):
        self.assertTrue(VERIFY_SYSTEM_PROMPT.startswith(CASE_A_VERIFY_MARKER))
        mock = MockInferenceClient()
        self.assertEqual(mock.generate(build_verify_prompt(Q, CLEAN))["text"], "YES")
        self.assertEqual(mock.generate(build_verify_prompt(Q, "this is a logic mistake"))["text"], "NO")
        self.assertEqual(mock.generate(build_verify_prompt(Q, "this is a wording mistake"))["text"], "NO")
        # No collisions: each marker appears in its own system prompt only.
        verify_prompt = build_verify_prompt(Q, CLEAN)
        classify_prompt = build_classifier_prompt(CLEAN, Q)
        self.assertNotIn(CLASSIFIER_PROMPT_MARKER, verify_prompt)
        self.assertNotIn(HINT_PROMPT_MARKER, verify_prompt)
        self.assertNotIn(CASE_A_VERIFY_MARKER, classify_prompt)
        self.assertNotIn(CASE_A_VERIFY_MARKER, SYSTEM_PROMPT)
        self.assertNotIn(CASE_A_VERIFY_MARKER, HINT_SYSTEM_PROMPT)
        # The classify prompt still gets a LABEL reply, not YES/NO.
        self.assertTrue(mock.generate(classify_prompt)["text"].startswith("LABEL:"))

    def test_case_a_verified_logged(self):
        s = self._session(make_verify_fn(MockInferenceClient()))
        s.submit_answer(CLEAN)
        self.assertIs(self._events(s, "answer")[-1]["case_a_verified"], True)

        s = self._session(make_verify_fn(MockInferenceClient()))
        s.submit_answer(NATURAL)  # Case C: verification never runs
        answer = self._events(s, "answer")[-1]
        self.assertEqual(answer["error_source"], "llm")
        self.assertIsNone(answer["case_a_verified"])

        s = self._session(None)  # no verify_fn: Case A is not verified
        s.submit_answer(CLEAN)
        answer = self._events(s, "answer")[-1]
        self.assertEqual((answer["error_source"], answer["case_a_verified"]), ("key_terms", None))


if __name__ == "__main__":
    unittest.main()
