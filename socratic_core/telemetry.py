"""
telemetry.py — aggregate counters over a session's answer events.

Role in the architecture
------------------------
Pure readers over ``SessionState.history``. Every function looks only at
``event == "answer"`` entries and reads each field through ``.get()``, so a
log persisted before a field existed (``matched_bank_id``,
``case_a_verified``, ``cloud_fallback_used``) counts that field as None
instead of raising. No I/O.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

ANSWER_EVENT = "answer"
# classifier_llm.MATCHED_NONE, restated so this module needs only the stdlib.
MATCHED_NONE = "none"


def _answers(turns: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [t for t in turns if t.get("event") == ANSWER_EVENT]


def bank_usage(turns: Iterable[Mapping[str, Any]]) -> Counter:
    """How often each few-shot example was matched, keyed by example ID (str).

    Skips turns that never reached the LLM (None) and turns where it matched
    nothing ("none"). Empty input -> ``Counter()``.
    """
    ids = [t.get("matched_bank_id") for t in _answers(turns)]
    return Counter(str(i) for i in ids if i is not None and i != MATCHED_NONE)


def case_a_failure_rate(turns: Iterable[Mapping[str, Any]]) -> float:
    """Fraction of verified Case A turns that verification rejected.

    Only turns with ``case_a_verified`` True or False count; 0.0 if none do.
    """
    verified = [t.get("case_a_verified") for t in _answers(turns)]
    verified = [v for v in verified if v is not None]
    if not verified:
        return 0.0
    return sum(1 for v in verified if v is False) / len(verified)


def cloud_fallback_rate(turns: Iterable[Mapping[str, Any]]) -> float:
    """Fraction of answer events decided by the Groq fallback; 0.0 with no answers."""
    answers = _answers(turns)
    if not answers:
        return 0.0
    return sum(1 for t in answers if t.get("cloud_fallback_used") is True) / len(answers)
