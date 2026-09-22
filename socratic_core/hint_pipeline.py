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

from typing import Any, Optional, Sequence

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
    "partial": (
        "The student named some of the key ideas but left others out. "
        "Ask a question that draws out the specific idea they omitted, "
        "rather than challenging the reasoning they already have right."
    ),
}

# Used when the model will not name the omitted idea itself. Phrased as a
# question so it clears the same validator as a generated hint.
PARTIAL_HINT_TEMPLATE = (
    "You're close, but your answer is missing the idea of {term}. "
    "What role does it play?"
)



def _clean_terms(missing_terms: Optional[Sequence[str]]) -> list[str]:
    return [t.strip() for t in (missing_terms or []) if t and t.strip()]


def names_a_missing_term(hint: str, missing_terms: Sequence[str]) -> bool:
    haystack = normalize(hint)
    return any(normalize(term) in haystack for term in missing_terms)


def generate_hint(
    question: Any,
    answer: str,
    error_type: str,
    client: Any,
    *,
    missing_terms: Optional[Sequence[str]] = None,
) -> str:
    """Ask the model for a hint. Returns its text, or "" on client error.

    ``missing_terms`` switches the request to the "partial" strategy: the
    student's reasoning is not what needs correcting, the gap is. The terms
    themselves are named in the prompt, which stays inside the module's
    invariant -- key_terms are a separate bank field from ``correct_answer``,
    ``accepted_variants`` and ``answer_explanation``, none of which are sent.
    """
    missing = _clean_terms(missing_terms)
    strategy_key = "partial" if missing else error_type
    lines = [
        f"Question: {question.question_text}",
        f"Student answer: {answer.strip()}",
        f"Error type: {strategy_key}",
    ]
    if missing:
        lines.append(f"Ideas the student left out: {', '.join(missing)}")
    lines.append(HINT_STRATEGY.get(strategy_key, HINT_STRATEGY["logic_error"]))
    result = client.generate(build_prompt("\n".join(lines), system=SYSTEM_PROMPT), max_tokens=MAX_TOKENS)
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


def hint_pipeline(
    question: Any,
    answer: str,
    error_type: str,
    client: Any,
    *,
    missing_terms: Optional[Sequence[str]] = None,
) -> dict:
    """Generate, validate, retry once, else fall back to a template.

    With ``missing_terms`` the bar is higher: a hint that clears the validator
    but names none of the omitted ideas is rejected too, because a partial
    answer needs pointing at the gap. When the model will not do that, the
    deterministic ``PARTIAL_HINT_TEMPLATE`` does -- and is itself validated,
    so the partial path can never leak what the model path could not.
    """
    missing = _clean_terms(missing_terms)
    rejections: list[str] = []
    for _ in range(2):
        hint = generate_hint(question, answer, error_type, client, missing_terms=missing)
        if not hint:
            break
        ok, reason = validate_hint(hint, question)
        if not ok:
            rejections.append(reason)
            continue
        if missing and not names_a_missing_term(hint, missing):
            rejections.append("names no missing key term")
            continue
        return {"hint": hint, "source": "llm", "rejections": rejections}

    if missing:
        templated = PARTIAL_HINT_TEMPLATE.format(term=missing[0])
        if validate_hint(templated, question)[0]:
            return {"hint": templated, "source": "partial_template", "rejections": rejections}
    return {"hint": question.fallback_hint, "source": "template", "rejections": rejections}
