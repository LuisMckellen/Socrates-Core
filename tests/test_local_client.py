"""Tests for local_client. The shape test needs the GGUF on disk and is
skipped otherwise; the missing-model test always runs.

Run from the project root:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# LocalCPUClient imports llama_cpp lazily, so the module below loads fine
# without it -- but generating anything needs the real runtime. Raised as
# unittest.SkipTest rather than pytest.importorskip so that the discover
# runner named in the docstring above skips this module too instead of
# erroring on a pytest-only exception type.
try:
    import llama_cpp  # noqa: F401
except ImportError:  # pragma: no cover - environment-dependent
    raise unittest.SkipTest("llama-cpp-python not installed in this environment")

from socratic_core.local_client import DEFAULT_MODEL_PATH, LocalCPUClient  # noqa: E402

MODEL_PATH = ROOT / DEFAULT_MODEL_PATH
RESULT_KEYS = {"text", "ttft_ms", "total_ms", "tokens_generated", "error"}


class LocalClientTests(unittest.TestCase):
    @unittest.skipUnless(os.path.isfile(MODEL_PATH), "model not downloaded")
    def test_local_client_returns_dict_shape(self):
        client = LocalCPUClient(str(MODEL_PATH))
        r = client.generate("Reply with the single word: hello", max_tokens=8)
        self.assertEqual(set(r), RESULT_KEYS)
        self.assertIsNone(r["error"])
        self.assertTrue(r["text"])
        self.assertGreater(r["total_ms"], 0.0)
        self.assertGreater(r["tokens_generated"], 0)

    def test_local_client_handles_missing_model(self):
        client = LocalCPUClient("models/does-not-exist.gguf")
        r = client.generate("hello")
        self.assertEqual(set(r), RESULT_KEYS)
        self.assertIsNotNone(r["error"])
        self.assertIn("not found", r["error"])
        self.assertEqual(r["text"], "")


if __name__ == "__main__":
    unittest.main()
