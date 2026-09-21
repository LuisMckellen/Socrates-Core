"""Offline tests for mastery.py and escalation.py.

Run from the project root:  python -m unittest discover -s tests -v
Expected values are derived from the constants in mastery.py; this file
must not restate any of them as a literal.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from socratic_core.escalation import choose_escalation  # noqa: E402
from socratic_core.mastery import (  # noqa: E402
    ESCALATION_THRESHOLD,
    FILTER_CORRECT_DELTA,
    FILTER_WRONG_DELTA,
    INITIAL_MASTERY,
    L2_MASTERY_THRESHOLD,
    MASTERY_CEILING,
    MASTERY_FLOOR,
    SOCRATIC_CORRECT_DELTA,
    SOCRATIC_WRONG_DELTA,
    level_for_mastery,
    should_escalate,
    update_mastery,
)
from socratic_core.question_bank import Question, QuestionBank  # noqa: E402

F_A = Question("f_001", "A", "fa?", "a", tier="filter", cluster="a")
S_A1 = Question("s_001", "A", "sa1?", "a", tier="socratic", cluster="a", fallback_hint="?")
S_A2 = Question("s_002", "A", "sa2?", "a", tier="socratic", cluster="a", fallback_hint="?")
S_B = Question("s_003", "B", "sb?", "b", tier="socratic", cluster="b", fallback_hint="?")
S_C = Question("s_004", "C", "sc?", "c", tier="socratic", cluster="c", fallback_hint="?")

# Level-aware fixtures for Change 1: two levels in one cluster.
S_A_L1 = Question("s_005", "A", "sa_l1?", "a", tier="socratic", cluster="a", fallback_hint="?", level=1)
S_A_L2 = Question("s_006", "A", "sa_l2?", "a", tier="socratic", cluster="a", fallback_hint="?", level=2)


class ChooseEscalationTests(unittest.TestCase):
    def test_no_candidates_returns_none(self):
        self.assertIsNone(choose_escalation(F_A, QuestionBank([F_A]), {}))
        self.assertIsNone(choose_escalation(F_A, QuestionBank([]), {}))

    def test_single_target_returns_it(self):
        bank = QuestionBank([F_A, S_A1, S_B])
        self.assertIs(choose_escalation(F_A, bank, {}), S_A1)
        # A same-cluster singleton wins even when another cluster is weaker.
        self.assertIs(choose_escalation(F_A, bank, {"b": MASTERY_FLOOR}), S_A1)

    def test_multiple_targets_picks_lowest_mastery(self):
        # Same-cluster candidates share one mastery score, so the lowest-mastery
        # rule degenerates to the tie-break: bank order, first one wins.
        bank = QuestionBank([F_A, S_A2, S_A1, S_B])
        self.assertIs(choose_escalation(F_A, bank, {}), S_A2)
        self.assertIs(choose_escalation(F_A, bank, {"a": MASTERY_FLOOR, "b": MASTERY_FLOOR}), S_A2)
        # The same rule applied across clusters does discriminate.
        bank = QuestionBank([F_A, S_B, S_C])
        self.assertIs(choose_escalation(F_A, bank, {"b": MASTERY_CEILING, "c": MASTERY_FLOOR}), S_C)

    def test_global_fallback_when_cluster_empty(self):
        bank = QuestionBank([F_A, S_B, S_C])
        # No cluster-a Socratic entry: lowest mastery anywhere; unknown clusters read as initial.
        self.assertIs(choose_escalation(F_A, bank, {"b": MASTERY_FLOOR}), S_B)
        self.assertIs(choose_escalation(F_A, bank, {"b": MASTERY_CEILING}), S_C)
        self.assertIs(choose_escalation(F_A, bank, {}), S_B)  # tie -> bank order

    def test_level_selection_low_mastery_picks_L1(self):
        bank = QuestionBank([F_A, S_A_L1, S_A_L2])
        below = math.nextafter(L2_MASTERY_THRESHOLD, MASTERY_FLOOR)
        self.assertEqual(level_for_mastery(below), 1)
        self.assertIs(choose_escalation(F_A, bank, {"a": below}), S_A_L1)

    def test_level_selection_high_mastery_picks_L2(self):
        bank = QuestionBank([F_A, S_A_L1, S_A_L2])
        self.assertEqual(level_for_mastery(L2_MASTERY_THRESHOLD), 2)
        self.assertIs(choose_escalation(F_A, bank, {"a": L2_MASTERY_THRESHOLD}), S_A_L2)

    def test_level_selection_single_level_cluster(self):
        # Exactly one Socratic entry in the cluster wins regardless of level/mastery.
        bank = QuestionBank([F_A, S_A_L1, S_B])
        self.assertIs(choose_escalation(F_A, bank, {"a": MASTERY_CEILING}), S_A_L1)
        self.assertIs(choose_escalation(F_A, bank, {"a": MASTERY_FLOOR}), S_A_L1)

    def test_global_fallback_still_works(self):
        # No local Socratic entries at all: level logic never engages, existing rule 3 applies.
        bank = QuestionBank([F_A, S_B, S_C])
        self.assertIs(choose_escalation(F_A, bank, {"b": MASTERY_FLOOR, "c": MASTERY_CEILING}), S_B)
        self.assertIs(choose_escalation(F_A, bank, {"b": MASTERY_CEILING, "c": MASTERY_FLOOR}), S_C)


class MasteryTests(unittest.TestCase):
    def test_mastery_update_on_filter_correct(self):
        m = update_mastery({}, "a", "filter", True)
        self.assertAlmostEqual(m["a"], INITIAL_MASTERY + FILTER_CORRECT_DELTA)

    def test_mastery_update_on_filter_wrong(self):
        m = update_mastery({}, "a", "filter", False)
        self.assertAlmostEqual(m["a"], INITIAL_MASTERY + FILTER_WRONG_DELTA)

    def test_mastery_update_on_socratic_correct(self):
        m = update_mastery({}, "a", "socratic", True)
        self.assertAlmostEqual(m["a"], INITIAL_MASTERY + SOCRATIC_CORRECT_DELTA)

    def test_mastery_update_on_socratic_wrong(self):
        m = update_mastery({}, "a", "socratic", False)
        self.assertAlmostEqual(m["a"], INITIAL_MASTERY + SOCRATIC_WRONG_DELTA)

    def test_mastery_clamped_to_zero_and_one(self):
        m = {}
        for _ in range(10):
            update_mastery(m, "up", "socratic", True)
            update_mastery(m, "down", "socratic", False)
        self.assertEqual(m["up"], MASTERY_CEILING)
        self.assertEqual(m["down"], MASTERY_FLOOR)
        # Other clusters are untouched and the same dict is returned.
        self.assertIs(update_mastery(m, "up", "filter", True), m)
        self.assertEqual(set(m), {"up", "down"})

    def test_escalation_threshold_respected(self):
        self.assertFalse(should_escalate({}, "a"))  # initial mastery is above threshold
        self.assertFalse(should_escalate({"a": ESCALATION_THRESHOLD}, "a"))  # strict <
        self.assertTrue(should_escalate({"a": math.nextafter(ESCALATION_THRESHOLD, MASTERY_FLOOR)}, "a"))
        self.assertTrue(should_escalate({"a": MASTERY_FLOOR}, "a"))
        self.assertFalse(should_escalate({"a": MASTERY_CEILING}, "a"))
        # One wrong filter from a fresh cluster is enough to cross the line.
        self.assertTrue(should_escalate(update_mastery({}, "a", "filter", False), "a"))


if __name__ == "__main__":
    unittest.main()
