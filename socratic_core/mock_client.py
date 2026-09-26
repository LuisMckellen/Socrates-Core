"""
mock_client.py — offline stand-in for ``InferenceClient``.

Role in the architecture
------------------------
Lets ``classifier_llm`` and (later) ``hint_pipeline`` be exercised on an
x86-64 dev machine where the Genie runtime cannot run. Returns the same
five-key dict as ``InferenceClient.generate()`` so callers cannot tell the
difference, and records every call so tests can assert on what was sent.

Dispatch is on the opening phrase of each system prompt, which is unique to
its template: the hint prompt never says "grading" and the classifier prompt
never says "Socratic tutor". Inside the classifier branch the label is
chosen by whichever of ``correct``/``partial``/``wording``/``logic`` appears *last* in
the student's answer (the text after the final "Student answer:"), so a
test can steer the label by what it puts in the answer; no steering word
means ``logic_error``. ``NOISE_TOLERANCE_PREFIX`` is cut off first: it ends
in "present and correct." and would otherwise steer every answer to
``correct``. ``label_overrides`` maps an exact student answer to a steering
word and wins over the words in it, for tests that need an answer with no
steering word to take a given label. The UI does not use it.

The Case A verify prompt gets ``YES``, or ``NO`` when the answer's last
steering word is ``partial``, ``wording`` or ``logic`` (the same convention).

The marker strings are copied from ``hint_pipeline.SYSTEM_PROMPT``,
``classifier_llm.SYSTEM_PROMPT`` and ``classifier_llm.VERIFY_SYSTEM_PROMPT``.
If an opening sentence is reworded, update the matching marker here or the
mock falls through to the generic reply.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from .inference_client import DEFAULT_CHAT_TEMPLATE
from .noise import NOISE_TOLERANCE_PREFIX

HINT_PROMPT_MARKER = "You are a Socratic tutor."
CLASSIFIER_PROMPT_MARKER = "You are grading a student's answer for a tutor."
CASE_A_VERIFY_MARKER = "Does this student answer state the biological mechanism correctly"
_STUDENT_ANSWER_MARKER = "Student answer:"
# Closes the user block; everything before it after the marker is the answer.
_USER_SUFFIX = DEFAULT_CHAT_TEMPLATE["user_suffix"]
# Word-initial, so "incorrect" does not steer to correct.
_STEER = re.compile(r"\b(correct|partial|wording|logic)", re.IGNORECASE)

# Canned completions, in the same ``LABEL/REASONING`` layout that
# classifier_llm asks the real model for (no MATCHED line: matched_bank_id
# comes back "none").
_CORRECT_TEXT = (
    "LABEL: correct\n"
    "REASONING: The student states the mechanism accurately."
)
_PARTIAL_TEXT = (
    "LABEL: partial\n"
    "REASONING: The student states part of the mechanism but leaves the rest out."
)
_WORDING_TEXT = (
    "LABEL: wording_error\n"
    "REASONING: The student has the right idea but used the wrong term for it."
)
_LOGIC_TEXT = (
    "LABEL: logic_error\n"
    "REASONING: The student uses the right vocabulary but the reasoning does not hold."
)
_HINT_TEXT = "What would happen to that process if the part you named were removed?"
_GENERIC_TEXT = "Understood."


class MockInferenceClient:
    """Drop-in for ``InferenceClient`` with canned, keyword-driven replies."""

    def __init__(
        self,
        fail_mode: bool = False,
        label_overrides: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.fail_mode = fail_mode
        # exact student answer -> steering word ("correct", "partial", "wording", "logic")
        self.label_overrides: dict[str, str] = dict(label_overrides or {})
        self.call_log: list[dict[str, Any]] = []
        self.last_result: Optional[dict[str, Any]] = None

    def generate(self, prompt: str, max_tokens: int = 256) -> dict:
        if self.fail_mode:
            result = self._result("", error="mock failure")
        else:
            result = self._result(self._canned_text(prompt))
        self.call_log.append({"prompt": prompt, "response": result})
        self.last_result = result
        return result

    def _canned_text(self, prompt: str) -> str:
        # Checked first so a verify prompt can never be read as a classify one.
        if CASE_A_VERIFY_MARKER in prompt:
            steers = _STEER.findall(self._student_answer(prompt))
            return "NO" if steers and steers[-1].lower() in ("partial", "wording", "logic") else "YES"
        if HINT_PROMPT_MARKER in prompt:
            return _HINT_TEXT
        if CLASSIFIER_PROMPT_MARKER in prompt:
            answer = self._student_answer(prompt)
            last = self.label_overrides.get(answer.split(_USER_SUFFIX, 1)[0].strip())
            if last is None:
                steers = _STEER.findall(answer)
                last = steers[-1].lower() if steers else "logic"
            return {"correct": _CORRECT_TEXT, "partial": _PARTIAL_TEXT, "wording": _WORDING_TEXT}.get(last, _LOGIC_TEXT)
        return _GENERIC_TEXT

    @staticmethod
    def _student_answer(prompt: str) -> str:
        """The case's student answer, with the noise-tolerance prefix removed."""
        answer = prompt.rsplit(_STUDENT_ANSWER_MARKER, 1)[-1]
        if NOISE_TOLERANCE_PREFIX in answer:
            answer = answer.split(NOISE_TOLERANCE_PREFIX, 1)[1]
        return answer

    @staticmethod
    def _result(text: str, error: Optional[str] = None) -> dict[str, Any]:
        tokens = len(text.split())
        return {
            "text": text,
            "ttft_ms": 0.0 if error else 1.0,
            "total_ms": 0.0 if error else float(tokens),
            "tokens_generated": tokens,
            "error": error,
        }
