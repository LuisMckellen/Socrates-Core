"""Offline tests for the Groq client and the classifier-failure cloud fallback.

No network: Groq is never called. The fallback is exercised with plain
callables in the ``classifier_fn`` shape.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from socratic_core import cloud_client  # noqa: E402
from socratic_core.cloud_client import API_KEY_NAME, GroqClient, GroqConfigError, split_chatml  # noqa: E402
from socratic_core.classifier_llm import make_classifier_fn  # noqa: E402
from socratic_core.inference_client import build_prompt  # noqa: E402
from socratic_core.mock_client import MockInferenceClient  # noqa: E402
from socratic_core.question_bank import load_question_bank  # noqa: E402
from socratic_core.state_machine import CLOUD_FALLBACK_SOURCE, SocraticSession  # noqa: E402
from ui import state  # noqa: E402

BANK = load_question_bank(ROOT / "question_bank.json")
# Shares no key term with s_cell_theory_L1, so it reaches the LLM layer.
NATURAL = "The body makes more of itself when you get hurt"


def _no_key():
    """Neither st.secrets nor the environment carries a Groq key."""
    env = {k: v for k, v in os.environ.items() if k != API_KEY_NAME}
    return (
        patch.object(cloud_client, "_secret_key", return_value=None),
        patch.dict(os.environ, env, clear=True),
    )


def _label(error_type: str, failed: bool = False):
    """A classifier_fn; ``failed`` marks it as one of classify_llm's fail-closed fallbacks."""
    reply = {"error_type": error_type, "reasoning": "t"}
    if failed:
        reply["classifier_failed"] = True
    return lambda q, a: reply


class GroqClientTests(unittest.TestCase):
    def test_groq_client_raises_without_key(self):
        secrets, env = _no_key()
        with secrets, env, self.assertRaises(GroqConfigError) as cm:
            GroqClient()
        self.assertIn(API_KEY_NAME, str(cm.exception))
        with secrets, env:
            self.assertFalse(cloud_client.groq_available())

    def test_split_chatml_recovers_system_and_user(self):
        messages = split_chatml(build_prompt("the user turn\nline two", system="the system turn"))
        self.assertEqual(
            messages,
            [{"role": "system", "content": "the system turn"}, {"role": "user", "content": "the user turn\nline two"}],
        )
        self.assertEqual(split_chatml("plain text"), [{"role": "user", "content": "plain text"}])


class CloudFallbackTests(unittest.TestCase):
    def setUp(self):
        self.sessions_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def _session(self, classifier_fn, fallback_classifier_fn=None):
        return SocraticSession(
            BANK, question_ids=["s_cell_theory_L1"], classifier_fn=classifier_fn,
            fallback_classifier_fn=fallback_classifier_fn, hint_fn=lambda q, a, e: "hint?",
            sessions_dir=self.sessions_dir,
        )

    @staticmethod
    def _last(s, event):
        return [h for h in s.state.history if h["event"] == event][-1]

    def test_local_failure_triggers_fallback(self):
        groq = Mock(side_effect=_label("correct"))
        s = self._session(_label("logic_error", failed=True), groq)
        r = s.submit_answer(NATURAL)
        groq.assert_called_once()
        self.assertEqual((r.kind, r.error_source), ("correct", CLOUD_FALLBACK_SOURCE))
        answer = self._last(s, "answer")
        self.assertEqual((answer["error_source"], answer["cloud_fallback_used"]), (CLOUD_FALLBACK_SOURCE, True))
        self.assertTrue(self._last(s, "llm_classify")["cloud_fallback_attempted"])

    def test_real_local_failure_triggers_fallback(self):
        # The path that is reachable in practice: classify_llm's own fail-closed result.
        groq = Mock(side_effect=_label("wording_error"))
        s = self._session(make_classifier_fn(MockInferenceClient(fail_mode=True)), groq)
        r = s.submit_answer(NATURAL)
        groq.assert_called_once()
        self.assertEqual((r.error_type, r.error_source), ("wording_error", CLOUD_FALLBACK_SOURCE))

    def test_successful_local_skips_fallback(self):
        groq = Mock(side_effect=_label("correct"))
        s = self._session(_label("logic_error"), groq)
        r = s.submit_answer(NATURAL)
        groq.assert_not_called()
        self.assertEqual((r.error_type, r.error_source), ("logic_error", "llm"))
        self.assertFalse(self._last(s, "answer")["cloud_fallback_used"])

    def test_local_error_triggers_fallback(self):
        s = self._session(Mock(side_effect=RuntimeError("llama died")), _label("wording_error"))
        r = s.submit_answer(NATURAL)
        self.assertEqual((r.error_type, r.error_source), ("wording_error", CLOUD_FALLBACK_SOURCE))
        self.assertTrue(self._last(s, "answer")["cloud_fallback_used"])

    def test_failed_fallback_keeps_local_result(self):
        for groq in (Mock(side_effect=ConnectionError("offline")), _label("correct", failed=True)):
            with self.subTest(groq=groq):
                s = self._session(_label("logic_error", failed=True), groq)
                r = s.submit_answer(NATURAL)
                self.assertEqual((r.error_type, r.error_source), ("logic_error", "llm"))
                answer = self._last(s, "answer")
                self.assertFalse(answer["cloud_fallback_used"])
                self.assertNotIn("confidence", answer)
                self.assertTrue(self._last(s, "llm_classify")["classifier_failed"])

    def test_fallback_disabled_when_no_groq_key(self):
        secrets, env = _no_key()
        with secrets, env:
            self.assertIsNone(state.cloud_fallback_fn(True, state.BACKEND_LOCAL))
        # Opt-in and backend gate even before the key is looked at.
        self.assertIsNone(state.cloud_fallback_fn(False, state.BACKEND_LOCAL))
        self.assertIsNone(state.cloud_fallback_fn(True, state.BACKEND_MOCK))
        # And a session without a fallback keeps its failed local label.
        s = self._session(_label("logic_error", failed=True), None)
        r = s.submit_answer(NATURAL)
        self.assertEqual((r.error_type, r.error_source), ("logic_error", "llm"))
        self.assertFalse(self._last(s, "answer")["cloud_fallback_used"])
        self.assertNotIn("cloud_fallback_attempted", self._last(s, "llm_classify"))


if __name__ == "__main__":
    unittest.main()
