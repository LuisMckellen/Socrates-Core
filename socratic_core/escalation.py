"""
escalation.py — pick the Socratic question to ask after a failed filter.

Pure function over the bank and the mastery dict. Whether to escalate at
all is the caller's decision (``mastery.should_escalate``); this module
only answers *which* question, via a fixed fallback chain:

    1. two or more Socratic entries in the failed filter's cluster
       -> the one at the mastery-appropriate level (``mastery.level_for_mastery``),
          picking the lowest-mastery cluster among those, or falling back to
          the closest level (the other one, since only 1/2 exist) if none
          match the target level (bank order on ties either way)
    2. exactly one Socratic entry in that cluster -> it, regardless of level
    3. any Socratic entry anywhere -> the one whose cluster has the lowest
       mastery (bank order on ties)
    4. none -> None; the caller logs a topic gap and moves on

Steps 1 and 3 share one rule: ``min`` by cluster mastery. Python's ``min``
keeps the first of equal keys, which is exactly the bank-order tie-break.
"""

from __future__ import annotations

from typing import Optional

from .mastery import INITIAL_MASTERY, level_for_mastery
from .question_bank import Question, QuestionBank


def choose_escalation(
    failed_question: Question, bank: QuestionBank, mastery: dict
) -> Optional[Question]:
    socratic = [q for q in bank if q.tier == "socratic"]
    local = [q for q in socratic if q.cluster == failed_question.cluster]

    if len(local) >= 2:
        target_level = level_for_mastery(mastery.get(failed_question.cluster, INITIAL_MASTERY))
        at_level = [q for q in local if q.level == target_level]
        candidates = at_level or local  # else the closest (only other) level
    elif len(local) == 1:
        return local[0]
    elif socratic:
        candidates = socratic
    else:
        return None
    return min(candidates, key=lambda q: mastery.get(q.cluster, INITIAL_MASTERY))
