"""Offline tests for synonyms.py (Change 4: key_terms synonym expansion).

Run from the project root:  python -m unittest discover -s tests -v
No model, no NPU, no network.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from socratic_core.synonyms import SYNONYM_MAP, expand_terms  # noqa: E402


class SynonymMapTests(unittest.TestCase):
    def test_synonym_map_expands_correctly(self):
        self.assertEqual(SYNONYM_MAP["cell"], ["cells"])
        self.assertIn("split", expand_terms("divide"))
        self.assertIn("splits", expand_terms("divide"))
        self.assertIn("divide", expand_terms("split"))
        self.assertIn("digesting", expand_terms("digest"))
        self.assertIn("digest", expand_terms("digesting"))
        self.assertIn("preexisting", expand_terms("pre-existing"))
        self.assertIn("pre-existing", expand_terms("preexisting"))
        self.assertIn("cells", expand_terms("cell"))
        self.assertIn("cell", expand_terms("cells"))

    def test_expand_terms_returns_original_when_not_in_map(self):
        self.assertEqual(expand_terms("mitochondria"), ["mitochondria"])
        self.assertEqual(expand_terms("unknown_term_xyz"), ["unknown_term_xyz"])
        # The term itself is always first, regardless of whether it has synonyms.
        self.assertEqual(expand_terms("digest")[0], "digest")


if __name__ == "__main__":
    unittest.main()
