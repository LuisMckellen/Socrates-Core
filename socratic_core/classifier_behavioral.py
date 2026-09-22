"""
classifier_behavioral.py — layer 1 of error classification (pure Python, 0 ms).

Role in the architecture
------------------------
Runs *before* the bank misconception lookup and before any NPU call. It only
ever produces one verdict, ``low_effort``, for answers that are obviously not
a real attempt. Everything else returns ``None``, which the state machine
reads as "continue to the next layer".

Rules, in this exact order:
  1. give-up keyword ("idk", "skip", "don't know", ...)   -> low_effort
     Highest priority: "idk I didn't study this" is long enough to pass a
     length check and would otherwise slip through to the LLM.
  2. word_count < question.min_words                      -> low_effort
     Lower bound only. There is no maximum length; a long answer is only
     wrong if its content is wrong.
  3. no keyword overlap with the question's vocabulary    -> routing signal
     Off-topic on its own is *not* a verdict. A student paraphrasing in their
     own words ("The body makes more of itself when you get hurt") shares no
     stemmed vocabulary with the question, yet is a genuine attempt that the
     LLM should judge. Only off-topic *paired with* a disengagement token
     ("skibidi", "tung", ...) is treated as low_effort; off-topic alone
     returns None and falls through to key_terms and then the LLM.
  4. otherwise                                            -> None

The state machine checks correctness before calling this, so a short correct
answer never reaches rule 2.

This module never calls the model and must stay that way — it is what makes
a large share of turns free.
"""

from __future__ import annotations

import re
from typing import Optional

from .disengagement import flag_disengagement
from .question_bank import ErrorType, Question, crude_stem, normalize, tokenize

# Answers with fewer content words than this are too short for the
# off-topic rule to be meaningful.
OFF_TOPIC_MIN_CONTENT_WORDS: int = 2

# Matched as whole words/phrases against the normalised answer.
LOW_EFFORT_PHRASES: tuple[str, ...] = (
    "idk",
    "i dont know",
    "i don't know",
    "dont know",
    "don't know",
    "dunno",
    "i dunno",
    "no idea",
    "no clue",
    "skip",
    "whatever",
    "who cares",
    "next question",
)

_PHRASE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(rf"(?<![a-z0-9']){re.escape(p)}(?![a-z0-9'])") for p in LOW_EFFORT_PHRASES
)


def contains_give_up_phrase(answer: str) -> Optional[str]:
    """Return the matched phrase, or None."""
    text = normalize(answer)
    for phrase, pat in zip(LOW_EFFORT_PHRASES, _PHRASE_PATTERNS):
        if pat.search(text):
            return phrase
    return None


def word_count(answer: str) -> int:
    return len(normalize(answer).split())


def is_too_short(answer: str, question: Question) -> bool:
    return word_count(answer) < question.min_words


def is_off_topic(answer: str, question: Question) -> bool:
    """True when the answer shares *no* stemmed content word with the question.

    Only fires for answers with at least ``OFF_TOPIC_MIN_CONTENT_WORDS``
    content words. Known misconceptions are part of the question vocabulary,
    so a bank-listed wrong answer is never judged off-topic here.
    """
    answer_words = {crude_stem(t) for t in tokenize(answer)}
    if len(answer_words) < OFF_TOPIC_MIN_CONTENT_WORDS:
        return False
    return answer_words.isdisjoint(question.keywords())


def explain(answer: str, question: Question) -> Optional[str]:
    """Same decision as ``classify_behavioural`` but returns *why* (for the log)."""
    phrase = contains_give_up_phrase(answer)
    if phrase:
        return f"give-up phrase: {phrase!r}"
    if is_too_short(answer, question):
        return f"fewer than min_words={question.min_words} words"
    if is_off_topic(answer, question):
        # Deliberate second call: state_machine already ran flag_disengagement
        # for the log. Reading it there instead would mean changing this
        # function's two-argument signature, which callers depend on.
        disengaged, tokens = flag_disengagement(answer)
        if disengaged:
            return f"off-topic with disengagement tokens: {tokens}"
        # Off-topic alone is a routing signal, not a verdict -> next layer.
    return None


def classify_behavioural(answer: str, question: Question) -> Optional[ErrorType]:
    """Return ``"low_effort"`` or ``None`` (meaning: continue to the next layer)."""
    return "low_effort" if explain(answer, question) else None


classify = classify_behavioural
