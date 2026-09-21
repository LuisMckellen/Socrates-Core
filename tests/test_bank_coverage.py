"""Offline C2 guard: every socratic accepted_variant must Case-A-match its
own key_terms (with synonym expansion). This is a guard on the live bank,
not a unit test of the classifier -- it never edits question_bank.json.

Run from the project root:  python -m unittest discover -s tests -v
No model, no NPU, no network.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from socratic_core.question_bank import load_question_bank  # noqa: E402
from socratic_core.state_machine import SocraticSession  # noqa: E402

BANK = load_question_bank(ROOT / "question_bank.json")


class BankCoverageTests(unittest.TestCase):
    def test_every_socratic_variant_produces_case_a(self):
        failures: list[tuple[str, str, list[str]]] = []
        for question in BANK:
            if question.tier != "socratic":
                continue
            for variant in question.accepted_variants:
                verdict, _matched, missing = SocraticSession._classify_by_key_terms(question, variant)
                if verdict != "correct":
                    failures.append((question.id, variant, missing))

        if failures:
            report = "\n".join(f"  {qid!r} variant={variant!r} missing={missing}" for qid, variant, missing in failures)
            self.fail(f"{len(failures)} accepted_variant(s) did not Case-A-match their key_terms:\n{report}")


if __name__ == "__main__":
    unittest.main()
