"""
mastery.py — per-cluster mastery score and the escalation threshold.

Every tuning value for the two-tier flow lives here and only here. Other
modules import what they need; none may restate a value as a literal.

Mastery is a float in [MASTERY_FLOOR, MASTERY_CEILING] per cluster. A
cluster the student has never touched reads as INITIAL_MASTERY. Filters
move it a little, Socratic questions move it a lot; a wrong filter answer
escalates only while the cluster is below ESCALATION_THRESHOLD.
"""

from __future__ import annotations

INITIAL_MASTERY = 0.50
FILTER_CORRECT_DELTA = +0.15
FILTER_WRONG_DELTA = -0.25
SOCRATIC_CORRECT_DELTA = +0.30
SOCRATIC_PARTIAL_DELTA = -0.10
SOCRATIC_WRONG_DELTA = -0.30
MASTERY_FLOOR = 0.0
MASTERY_CEILING = 1.0
ESCALATION_THRESHOLD = 0.40
# Cluster mastery at/above this picks the level-2 Socratic follow-up on
# escalation; below it picks level 1. See escalation.choose_escalation.
L2_MASTERY_THRESHOLD = 0.60

_DELTAS: dict[tuple[str, object], float] = {
    ("filter", True): FILTER_CORRECT_DELTA,
    ("filter", False): FILTER_WRONG_DELTA,
    ("socratic", True): SOCRATIC_CORRECT_DELTA,
    ("socratic", False): SOCRATIC_WRONG_DELTA,
    ("socratic", "partial"): SOCRATIC_PARTIAL_DELTA,
}


def update_mastery(mastery: dict, cluster: str, tier: str, correct: bool | str) -> dict:
    """Apply the delta for (tier, correct) to ``cluster`` in place and return the dict.

    ``correct`` is normally a bool; pass the literal string ``"partial"`` for a
    Socratic key-terms partial match (see ``state_machine._classify_by_key_terms``).
    """
    try:
        delta = _DELTAS[(tier, correct)]
    except KeyError:
        raise ValueError(f"unknown tier {tier!r}") from None
    current = mastery.get(cluster, INITIAL_MASTERY)
    mastery[cluster] = max(MASTERY_FLOOR, min(MASTERY_CEILING, current + delta))
    return mastery


def should_escalate(mastery: dict, cluster: str) -> bool:
    return mastery.get(cluster, INITIAL_MASTERY) < ESCALATION_THRESHOLD


def level_for_mastery(mastery: float) -> int:
    """Which Socratic level a cluster's mastery score should escalate to."""
    return 2 if mastery >= L2_MASTERY_THRESHOLD else 1
