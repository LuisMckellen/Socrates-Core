"""
noise.py — meme/slang tolerance prompt prefix for the LLM classifier (Change 4).

Role in the architecture
------------------------
``state_machine.py`` prepends ``NOISE_TOLERANCE_PREFIX`` to the answer text
it hands to ``classifier_fn`` so the underlying prompt (built inside
``classifier_llm.py``, which this module does not touch) tells the model to
judge the biology, not the noise. Disengagement detection (``disengagement.py``)
is a separate, purely-logged signal and never feeds this prefix.
"""

from __future__ import annotations

NOISE_TOLERANCE_PREFIX = (
    "The student's answer may contain meme tokens, slang, or off-topic "
    "language mixed with real content. Ignore the noise. Judge only "
    "whether the biological concept is present and correct."
)
