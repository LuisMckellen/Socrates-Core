"""
question_bank.py — load and query the Socratic question bank.

Role in the architecture
------------------------
Pure data layer. Loads ``question_bank.json`` into typed dataclasses and
offers 0 ms lookups that the state machine and the behavioural classifier
rely on:

* ``Question.is_correct(answer)``          — exact match after normalisation
* ``Question.match_misconception(answer)`` — known wrong answer -> error type
* ``Question.keywords()``                  — topic vocabulary for off-topic checks

Nothing here touches the model. The bank is the *only* place the correct
answer lives; the LLM never sees it except inside the hint validator
(pure Python) which checks that a hint does not leak it.

Path resolution (no hardcoded paths): pass ``path`` explicitly, or set the
``SOCRATIC_QUESTION_BANK`` environment variable, or fall back to
``question_bank.json`` in the project root (the parent of this package).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal, Optional

ErrorType = Literal["correct", "wording_error", "logic_error", "low_effort"]
VALID_ERROR_TYPES: frozenset[str] = frozenset(
    {"correct", "wording_error", "logic_error", "low_effort"}
)
# low_effort is a behavioural verdict, never a bank-authored misconception.
# correct and wording_error are runtime-only labels from the LLM classifier;
# the bank itself may only author logic_error misconceptions (CHECK 9).
BANK_ERROR_TYPES: frozenset[str] = frozenset({"logic_error"})

Tier = Literal["filter", "socratic"]
VALID_TIERS: frozenset[str] = frozenset({"filter", "socratic"})
SOCRATIC_MISCONCEPTION_RANGE = (1, 3)
SOCRATIC_LEVELS: frozenset[int] = frozenset({1, 2})
ID_PREFIX: dict[str, str] = {"filter": "f_", "socratic": "s_"}
MIN_WORDS_RANGE: dict[str, tuple[int, int]] = {"filter": (1, 2), "socratic": (3, 8)}
SOCRATIC_KEY_TERMS_RANGE = (1, 4)
_CLUSTER_FORMAT = re.compile(r"^[a-z][a-z0-9_]*$")

# Misconception ids: lowercase snake_case, unique across the whole bank,
# cluster-prefixed and naming the student's belief ("ct_hypertrophy"). They
# are the classifier's few-shot example IDs, so they must not collide with
# the other example IDs, "none", or the old positional m_0, m_1, ... still
# found in session logs written before stable ids existed.
_MISCONCEPTION_ID_FORMAT = re.compile(r"[a-z][a-z0-9_]*")
_LEGACY_POSITIONAL_ID = re.compile(r"m_\d+")
RESERVED_MISCONCEPTION_IDS: frozenset[str] = frozenset({"natural_correct", "partial_example", "none"})

ENV_QUESTION_BANK = "SOCRATIC_QUESTION_BANK"
DEFAULT_BANK_FILENAME = "question_bank.json"
DEFAULT_MIN_WORDS = 3

# Leading filler a student might type before the actual answer.
# Stripped before comparison so "i think it's the cell" == "cell".
_LEADING_FILLER = re.compile(
    r"^(?:(?:i think|i guess|maybe|probably|it is|its|it's|the answer is|"
    r"answer is|answer|is it|um+|uh+)\s+)+"
)
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+")
_NON_WORD = re.compile(r"[^a-z0-9'\s]")
_WHITESPACE = re.compile(r"\s+")

# Small stopword list used when building topic keyword sets.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the of to in on at for from by with and or but is are was were be
    been being do does did what which who whom whose where when why how
    this that these those it its they them their there here into onto as
    if then than so not no yes all any some each every both either neither
    can could may might must shall should will would about according i you
    we he she my your our
    """.split()
)


def normalize(text: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace.

    Used for both correctness checks and misconception matching so that
    "The Cell." and "cell" compare equal. Apostrophes are kept so that
    "don't" survives for the behavioural classifier.
    """
    text = text.lower().replace("’", "'")
    text = _NON_WORD.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def normalize_answer(text: str) -> str:
    """``normalize`` plus removal of leading filler and articles."""
    text = normalize(text)
    text = _LEADING_FILLER.sub("", text)
    text = _LEADING_ARTICLE.sub("", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Content words of ``text`` (normalised, stopwords removed)."""
    return [t for t in normalize(text).split() if t not in STOPWORDS]


def crude_stem(token: str) -> str:
    """Very small suffix stripper so 'cells' ~ 'cell', 'divided' ~ 'divid'.

    Deliberately naive: it only needs to make keyword overlap tolerant of
    plurals and tense, not be linguistically correct.
    """
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


@dataclass(frozen=True)
class Misconception:
    # Stable, semantic, bank-wide unique id (see _MISCONCEPTION_ID_FORMAT).
    id: str
    wrong_answer: str
    error_type: ErrorType
    explanation: str

    def matches(self, answer: str) -> bool:
        return normalize_answer(answer) == normalize_answer(self.wrong_answer)


@dataclass(frozen=True)
class Question:
    id: str
    topic: str
    question_text: str
    correct_answer: str
    accepted_variants: tuple[str, ...] = ()
    misconceptions: tuple[Misconception, ...] = ()
    fallback_hint: str = ""
    # Answers with fewer words than this are low_effort (lower bound only;
    # there is deliberately no upper bound). Optional in the JSON.
    min_words: int = DEFAULT_MIN_WORDS
    # Optional in the JSON. Second line of the escalation reveal; if absent
    # the state machine synthesises a generic one.
    answer_explanation: str = ""
    # "filter": one-fact probe, exact match only, no classification layers.
    # "socratic": full pipeline. Missing in the JSON -> socratic.
    tier: str = "socratic"
    # Groups a filter with the Socratic entries it can escalate to.
    # Missing in the JSON -> same as topic.
    cluster: str = ""
    key_terms: tuple[str, ...] = ()
    # Socratic difficulty: 1 or 2. Always None for a filter (CHECK 7).
    level: Optional[int] = None
    # Socratic-only, optional in the JSON (filters must not carry the key).
    # One correct answer in everyday words; the LLM classifier shows it as a
    # ``correct`` few-shot example for this question. "" means none.
    natural_correct_example: str = ""

    def __post_init__(self) -> None:
        if not self.cluster:
            object.__setattr__(self, "cluster", self.topic)

    # -- correctness ------------------------------------------------------

    def accepted_answers(self) -> tuple[str, ...]:
        return (self.correct_answer, *self.accepted_variants)

    def is_correct(self, answer: str) -> bool:
        """Exact match (after normalisation) against answer or any variant.

        Deliberately *not* substring-based: "cell membrane" must not be
        accepted for "cell". Add variants to the bank instead.
        """
        candidate = normalize_answer(answer)
        if not candidate:
            return False
        return any(candidate == normalize_answer(a) for a in self.accepted_answers())

    def match_misconception(self, answer: str) -> Optional[Misconception]:
        for m in self.misconceptions:
            if m.matches(answer):
                return m
        return None

    # -- vocabulary for the behavioural classifier -------------------------

    def keywords(self) -> frozenset[str]:
        """Stemmed content words that count as 'on topic' for this question.

        Drawn from the topic slug, the question text, the accepted answers
        and the known wrong answers (a known misconception is on-topic even
        though it is wrong).
        """
        sources = [
            self.topic.replace("_", " "),
            self.question_text,
            *self.accepted_answers(),
            *(m.wrong_answer for m in self.misconceptions),
        ]
        words: set[str] = set()
        for src in sources:
            words.update(crude_stem(t) for t in tokenize(src))
        return frozenset(words)


class QuestionBankError(ValueError):
    """Raised when question_bank.json is missing or malformed."""


@dataclass
class QuestionBank:
    questions: list[Question] = field(default_factory=list)
    source_path: Optional[Path] = None

    def __post_init__(self) -> None:
        self._by_id: dict[str, Question] = {q.id: q for q in self.questions}
        if len(self._by_id) != len(self.questions):
            raise QuestionBankError("duplicate question ids in bank")

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self) -> Iterator[Question]:
        return iter(self.questions)

    def __contains__(self, question_id: str) -> bool:
        return question_id in self._by_id

    def get(self, question_id: str) -> Question:
        try:
            return self._by_id[question_id]
        except KeyError:
            raise KeyError(f"unknown question id: {question_id!r}") from None

    def ids(self) -> list[str]:
        return [q.id for q in self.questions]

    def by_topic(self, topic: str) -> list[Question]:
        return [q for q in self.questions if q.topic == topic]

    def topics(self) -> list[str]:
        seen: dict[str, None] = {}
        for q in self.questions:
            seen.setdefault(q.topic, None)
        return list(seen)


# -- loading --------------------------------------------------------------


def default_bank_path() -> Path:
    env = os.environ.get(ENV_QUESTION_BANK)
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / DEFAULT_BANK_FILENAME


def _parse_misconception(raw: dict, qid: str, idx: int) -> Misconception:
    where = f"question {qid!r} misconception[{idx}]"
    try:
        mid = raw["id"]
        wrong = str(raw["wrong_answer"])
        etype = str(raw["error_type"])
    except KeyError as e:
        raise QuestionBankError(f"{where}: missing field {e}") from None
    if not isinstance(mid, str) or not _MISCONCEPTION_ID_FORMAT.fullmatch(mid):
        raise QuestionBankError(f"{where}: id must be lowercase snake_case (got {mid!r})")
    if mid in RESERVED_MISCONCEPTION_IDS or _LEGACY_POSITIONAL_ID.fullmatch(mid):
        raise QuestionBankError(
            f"{where}: id {mid!r} is reserved (classifier example ids, 'none', and positional m_<n>)"
        )
    if etype not in BANK_ERROR_TYPES:
        raise QuestionBankError(
            f"{where}: error_type {etype!r} not in {sorted(BANK_ERROR_TYPES)}"
        )
    return Misconception(
        id=mid,
        wrong_answer=wrong,
        error_type=etype,  # type: ignore[arg-type]
        explanation=str(raw.get("explanation", "")),
    )


def _parse_question(raw: dict, idx: int) -> Question:
    qid = str(raw.get("id", f"<index {idx}>"))
    required = ("id", "topic", "question_text", "correct_answer", "fallback_hint")
    missing = [k for k in required if k not in raw]
    if missing:
        raise QuestionBankError(f"question {qid!r}: missing fields {missing}")

    tier = str(raw.get("tier", "socratic"))
    if tier not in VALID_TIERS:
        raise QuestionBankError(f"question {qid!r}: tier {tier!r} not in {sorted(VALID_TIERS)}")
    if not qid.startswith(ID_PREFIX[tier]):
        raise QuestionBankError(
            f"question {qid!r}: {tier} id must start with {ID_PREFIX[tier]!r} (got {qid!r})"
        )

    cluster = str(raw.get("cluster") or raw["topic"])
    if not _CLUSTER_FORMAT.match(cluster):
        raise QuestionBankError(
            f"question {qid!r}: cluster must be lowercase snake_case (got {cluster!r})"
        )

    # CHECK 7: level is required (1 or 2) for socratic, forbidden for filter.
    if tier == "filter":
        if "level" in raw:
            raise QuestionBankError(f"question {qid!r}: tier 'filter' must not have a level field")
        level: Optional[int] = None
    else:
        level = raw.get("level")
        if not isinstance(level, int) or isinstance(level, bool) or level not in SOCRATIC_LEVELS:
            raise QuestionBankError(
                f"question {qid!r}: tier 'socratic' requires level in {sorted(SOCRATIC_LEVELS)} (got {level!r})"
            )

    # natural_correct_example: forbidden for filter; optional string for socratic.
    if tier == "filter":
        if "natural_correct_example" in raw:
            raise QuestionBankError(
                f"question {qid!r}: tier 'filter' must not have a natural_correct_example field"
            )
        natural_correct_example = ""
    else:
        natural_correct_example = raw.get("natural_correct_example", "")
        if not isinstance(natural_correct_example, str):
            raise QuestionBankError(
                f"question {qid!r}: natural_correct_example must be a string "
                f"(got {type(natural_correct_example).__name__})"
            )

    accepted_variants = tuple(str(v) for v in raw.get("accepted_variants", []))
    if not accepted_variants:
        raise QuestionBankError(f"question {qid!r}: accepted_variants must contain at least one entry")

    fallback_hint = str(raw["fallback_hint"])
    if fallback_hint.strip() and not fallback_hint.strip().endswith("?"):
        # The hint validator (module 4) would reject it later; fail early.
        raise QuestionBankError(f"question {qid!r}: fallback_hint must end with '?'")
    if tier == "socratic" and not fallback_hint.strip():
        raise QuestionBankError(f"question {qid!r}: tier 'socratic' requires a fallback_hint")

    miscs = tuple(
        _parse_misconception(m, qid, i)
        for i, m in enumerate(raw.get("misconceptions", []))
    )
    key_terms = tuple(str(k) for k in raw.get("key_terms", []))
    if tier == "filter":
        if miscs:
            raise QuestionBankError(
                f"question {qid!r}: tier 'filter' must have no misconceptions (found {len(miscs)})"
            )
        if key_terms:
            raise QuestionBankError(
                f"question {qid!r}: tier 'filter' must have no key_terms (found {len(key_terms)})"
            )
    else:
        # CHECK 8: relaxed from "exactly 3" to a 1-3 range.
        misc_lo, misc_hi = SOCRATIC_MISCONCEPTION_RANGE
        if not misc_lo <= len(miscs) <= misc_hi:
            raise QuestionBankError(
                f"question {qid!r}: tier 'socratic' must have {misc_lo}-{misc_hi} "
                f"misconceptions (found {len(miscs)})"
            )
        kt_lo, kt_hi = SOCRATIC_KEY_TERMS_RANGE
        if not kt_lo <= len(key_terms) <= kt_hi:
            raise QuestionBankError(
                f"question {qid!r}: socratic key_terms must have {kt_lo}–{kt_hi} entries "
                f"(got {len(key_terms)})"
            )

    min_words = raw.get("min_words", DEFAULT_MIN_WORDS)
    if not isinstance(min_words, int) or isinstance(min_words, bool) or min_words < 1:
        raise QuestionBankError(f"question {qid!r}: min_words must be an integer >= 1")
    mw_lo, mw_hi = MIN_WORDS_RANGE[tier]
    if not mw_lo <= min_words <= mw_hi:
        raise QuestionBankError(
            f"question {qid!r}: {tier} min_words must be in [{mw_lo}, {mw_hi}] (got {min_words})"
        )
    return Question(
        id=str(raw["id"]),
        topic=str(raw["topic"]),
        question_text=str(raw["question_text"]),
        correct_answer=str(raw["correct_answer"]),
        accepted_variants=accepted_variants,
        misconceptions=miscs,
        fallback_hint=fallback_hint,
        min_words=min_words,
        answer_explanation=str(raw.get("answer_explanation", "")),
        tier=tier,
        cluster=cluster,
        key_terms=key_terms,
        level=level,
        natural_correct_example=natural_correct_example,
    )


def load_question_bank(path: str | os.PathLike[str] | None = None) -> QuestionBank:
    """Load and validate the bank. Raises ``QuestionBankError`` on bad input."""
    bank_path = Path(path) if path is not None else default_bank_path()
    if not bank_path.is_file():
        raise QuestionBankError(
            f"question bank not found at {bank_path} "
            f"(set {ENV_QUESTION_BANK} or pass path=)"
        )
    try:
        data = json.loads(bank_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise QuestionBankError(f"{bank_path}: invalid JSON: {e}") from None
    raw_questions = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(raw_questions, list):
        raise QuestionBankError(f"{bank_path}: top-level 'questions' list missing")
    questions: list[Question] = []
    first_index: dict[str, int] = {}
    misconception_owner: dict[str, str] = {}
    for i, raw in enumerate(raw_questions):
        q = _parse_question(raw, i)
        if q.id in first_index:
            raise QuestionBankError(
                f"duplicate id {q.id!r} found at indices {first_index[q.id]} and {i}"
            )
        first_index[q.id] = i
        for m in q.misconceptions:
            if m.id in misconception_owner:
                raise QuestionBankError(
                    f"duplicate misconception id {m.id!r} in questions "
                    f"{misconception_owner[m.id]!r} and {q.id!r}: misconception ids must be "
                    f"unique across the whole bank (telemetry.bank_usage counts by bare id)"
                )
            misconception_owner[m.id] = q.id
        questions.append(q)
    return QuestionBank(questions=questions, source_path=bank_path)
