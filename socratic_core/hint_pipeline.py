"""
hint_pipeline.py — generate a Socratic hint on the NPU, validate it in pure Python.

Role in the architecture
------------------------
Runs after classification. The model is asked for a single guiding question
tailored to the error type; a 0 ms validator then decides whether that
question is safe to show. Anything the validator rejects is retried once,
and if that fails too the bank's hand-written ``fallback_hint`` is used.

The load-bearing invariant
--------------------------
The model never receives the correct answer. Its prompt carries exactly
three things: the question text, the student's answer, and the classified
error type. ``correct_answer``, ``accepted_variants`` and
``answer_explanation`` stay in the bank. The validator, which *does* read
them, is plain string matching — no model involved — so a leaked answer can
only ever be caught, never produced.
"""

from __future__ import annotations

from typing import Any

from .inference_client import build_prompt
from .question_bank import normalize

MAX_HINT_WORDS = 25
MAX_TOKENS = 64

SYSTEM_PROMPT = (
    "You are a Socratic tutor. The student answered a question incorrectly. "
    "Reply with ONE guiding question that helps them find the answer themselves. "
    f"Rules: fewer than {MAX_HINT_WORDS} words; end with '?'; never state the answer; "
    "no explanations, no preamble."
)

HINT_STRATEGY: dict[str, str] = {
    "wording_error": (
        "The student has the right idea but used the wrong term. "
        "Ask a question that nudges them toward the precise term without saying it."
    ),
    "logic_error": (
        "The student used the right terms but the reasoning is broken. "
        "Ask a question that exposes the flaw through a counter-example or analogy."
    ),
    "low_effort": (
        "The student is disengaged. "
        "Ask a smaller, easier question that gets them started."
    ),
}


def generate_hint(question: Any, answer: str, error_type: str, client: Any) -> str:
    """Ask the model for a hint. Returns its text, or "" on client error."""
    user = (
        f"Question: {question.question_text}\n"
        f"Student answer: {answer.strip()}\n"
        f"Error type: {error_type}\n"
        f"{HINT_STRATEGY.get(error_type, HINT_STRATEGY['logic_error'])}"
    )
    result = client.generate(build_prompt(user, system=SYSTEM_PROMPT), max_tokens=MAX_TOKENS)
    if result.get("error") is not None:
        return ""
    return str(result.get("text", "")).strip()


def validate_hint(hint: str, question: Any) -> tuple[bool, str]:
    """Pure-Python gate: (True, "ok") or (False, reason). Never calls the model."""
    stripped = hint.strip()
    if not stripped:
        return False, "empty hint"

    # normalize() lowercases and strips punctuation, so "Semi-conservative"
    # is still caught when the hint writes it as "semi conservative".
    hint_norm = normalize(stripped)
    if normalize(question.correct_answer) in hint_norm:
        return False, "contains correct answer"
    for variant in question.accepted_variants:
        if normalize(variant) in hint_norm:
            return False, f"contains accepted variant: {variant!r}"

    words = len(stripped.split())
    if words > MAX_HINT_WORDS:
        return False, f"too long: {words} words > {MAX_HINT_WORDS}"
    if not stripped.endswith("?"):
        return False, "does not end with '?'"
    return True, "ok"


def hint_pipeline(question: Any, answer: str, error_type: str, client: Any) -> dict:
    """Generate, validate, retry once, else fall back to the bank's template."""
    rejections: list[str] = []
    for _ in range(2):
        hint = generate_hint(question, answer, error_type, client)
        if not hint:
            break
        ok, reason = validate_hint(hint, question)
        if ok:
            return {"hint": hint, "source": "llm", "rejections": rejections}
        rejections.append(reason)
    return {"hint": question.fallback_hint, "source": "template", "rejections": rejections}
