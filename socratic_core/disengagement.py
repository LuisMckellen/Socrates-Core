"""
disengagement.py — flag meme/slang tokens in a student answer (Change 6).

Role in the architecture
------------------------
Logged on every turn, both tiers. Option A/B hybrid: disengagement is
observational on its own. It becomes verdict-affecting only when paired with
an off-topic answer in the behavioural classifier (see
``classifier_behavioral.py``). Standalone occurrence still does not alter
verdict, error_type, or mastery — a meme-laden answer that still contains the
right concept scores exactly as it would without the noise (see ``noise.py``
for the layer that actually tolerates the noise).
"""

from __future__ import annotations

import re

DISENGAGEMENT_TOKENS: list[str] = [
    "tung",
    "sahur",
    "skibidi",
    "sigma",
    "rizz",
    "mewing",
    "gyatt",
    "ohio",
    "slay",
    "fanum",
    "griddy",
    "bussin",
]

_TOKEN_PATTERNS: dict[str, re.Pattern[str]] = {
    t: re.compile(rf"(?<![a-z0-9']){re.escape(t)}(?![a-z0-9'])") for t in DISENGAGEMENT_TOKENS
}


def flag_disengagement(answer: str) -> tuple[bool, list[str]]:
    """Whole-word scan for meme/slang tokens. Returns ``(flagged, matched_tokens)``."""
    lowered = answer.lower()
    matched = [t for t in DISENGAGEMENT_TOKENS if _TOKEN_PATTERNS[t].search(lowered)]
    return bool(matched), matched
