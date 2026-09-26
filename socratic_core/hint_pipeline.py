"""
hint_pipeline.py — generate a Socratic hint on the NPU, validate it in pure Python.

Role in the architecture
------------------------
Runs after classification. The model is asked for a single guiding question
tailored to the error type -- or, when the classifier matched one of the
question's bank misconceptions, aimed at that misconception; a 0 ms
validator then decides whether that question is safe to show. Anything the
validator rejects is retried once, and if that fails too the bank's
hand-written ``fallback_hint`` is used.

The load-bearing invariant
--------------------------
``correct_answer``, ``accepted_variants`` and ``answer_explanation`` never go
to the generator; they stay in the bank. ``natural_correct_example`` does,
only on the targeted path (a matched misconception), so the model can aim
at the gap between the mistake and the right idea. It is guarded by
validator rules 3a (the example verbatim) and 3b (any
``PARAPHRASE_NGRAM``-word run of content words shared with the example,
``correct_answer`` or an accepted variant). The validator is plain string
matching -- no model involved -- so a leaked answer can only ever be caught,
never produced. 3b does not catch a synonym-for-synonym rewrite. On the
targeted path rule 3c also rejects any hint naming a key term (the correct
mechanism, e.g. offered as the other half of an either/or).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .inference_client import build_prompt
from .question_bank import crude_stem, normalize, tokenize
from .synonyms import expand_terms

MAX_HINT_WORDS = 25
MAX_TOKENS = 64
# Rule 3b window. 4 catches both paraphrase probes in test_hint_pipeline and
# rejects none of the real Local CPU hints recorded there; 5 misses one probe.
PARAPHRASE_NGRAM = 4

SYSTEM_PROMPT = (
    "You are a Socratic tutor. The student answered a question incorrectly. "
    "Reply with ONE guiding question that helps them find the answer themselves. "
    f"Rules: fewer than {MAX_HINT_WORDS} words; end with '?'; never state the answer; "
    "no explanations, no preamble."
)

HINT_STRATEGY: dict[str, str] = {
    "wording_error": (
        "The student has the right idea but used the wrong term. "
        "Ask a question that nudges them toward the precise term without saying it."
    ),
    "logic_error": (
        "The student used the right terms but the reasoning is broken. "
        "Ask a question that exposes the flaw through a counter-example or analogy."
    ),
    "low_effort": (
        "The student is disengaged. "
        "Ask a smaller, easier question that gets them started."
    ),
    "partial": (
        "The student named some of the key ideas but left others out. "
        "Ask a question that draws out the specific idea they omitted, "
        "rather than challenging the reasoning they already have right."
    ),
    "misconception": (
        "The student's answer states a mistaken belief. Ask ONE question that pushes the "
        "student to extend their own stated belief: what it implies, what would have to "
        "be true for it to work, where it leads. When they follow it to its end, the gap "
        "should become visible to them. Do not offer alternatives. Do not present two "
        "options. Do not name the misconception. Do not state, imply, or describe the "
        "correct mechanism. Ask about ONE thing only."
    ),
}

# Rule 3c (targeted path only): the word forms a key term takes in a question,
# beyond its synonyms.expand_terms() entries and crude_stem() matches. Only
# irregular forms need an entry: "division" -> "divide" is neither a synonym nor
# a shared stem. Keyed by the bank's key_terms, lowercase. 0b fills in the rest.
KEY_TERM_FORMS: dict[str, list[str]] = {
    "division": ["divide", "divides", "dividing", "divided"],
}

# Used when the model will not name the omitted idea itself. Phrased as a
# question so it clears the same validator as a generated hint.
PARTIAL_HINT_TEMPLATE = (
    "You're close, but your answer is missing the idea of {term}. "
    "What role does it play?"
)



def _clean_terms(missing_terms: Optional[Sequence[str]]) -> list[str]:
    return [t.strip() for t in (missing_terms or []) if t and t.strip()]


def names_a_missing_term(hint: str, missing_terms: Sequence[str]) -> bool:
    haystack = normalize(hint)
    return any(normalize(term) in haystack for term in missing_terms)


def resolve_misconception(question: Any, matched_bank_id: Optional[str]) -> Any:
    """The question's own misconception with this bank id, else None.

    None for ``natural_correct``, ``partial_example``, ``"none"``, None, or an
    id that belongs to another question: those keep the untargeted prompt.
    """
    if not matched_bank_id:
        return None
    return next((m for m in getattr(question, "misconceptions", ()) if m.id == matched_bank_id), None)


def build_hint_prompt(
    question: Any,
    answer: str,
    error_type: str,
    *,
    missing_terms: Optional[Sequence[str]] = None,
    matched_bank_id: Optional[str] = None,
) -> str:
    """The generator prompt. Partial (missing terms) wins over a matched
    misconception; with neither, the untargeted prompt is unchanged from before 0c."""
    missing = _clean_terms(missing_terms)
    misconception = None if missing else resolve_misconception(question, matched_bank_id)
    if misconception is not None:
        # wrong_answer is deliberately not sent: with it in the prompt the model
        # put the bank's wording ("swell") into hints for students who never used
        # it (0c ablation). The student's own answer is the belief to extend; the
        # matched id only selects this prompt and is logged.
        lines = [
            f"Topic: {question.topic}",
            f"Question: {question.question_text}",
            f"Student answer: {answer.strip()}",
        ]
        # Optional fields: a line is omitted, never rendered as "None".
        goal = (getattr(misconception, "diagnostic_goal", None) or "").strip()
        if goal:
            lines.append(f"Diagnostic goal: {goal}")
        natural = (getattr(question, "natural_correct_example", "") or "").strip()
        if natural:
            lines.append(
                "Correct idea in everyday words (for your judgement only; never repeat, "
                f"paraphrase or confirm it): {natural}"
            )
        lines.append(HINT_STRATEGY["misconception"])
        return build_prompt("\n".join(lines), system=SYSTEM_PROMPT)

    strategy_key = "partial" if missing else error_type
    lines = [
        f"Question: {question.question_text}",
        f"Student answer: {answer.strip()}",
        f"Error type: {strategy_key}",
    ]
    if missing:
        lines.append(f"Ideas the student left out: {', '.join(missing)}")
    lines.append(HINT_STRATEGY.get(strategy_key, HINT_STRATEGY["logic_error"]))
    return build_prompt("\n".join(lines), system=SYSTEM_PROMPT)


def generate_hint(
    question: Any,
    answer: str,
    error_type: str,
    client: Any,
    *,
    missing_terms: Optional[Sequence[str]] = None,
    matched_bank_id: Optional[str] = None,
) -> Optional[str]:
    """Ask the model for a hint. Returns its text ("" for an empty reply), or
    None on client error.

    ``missing_terms`` switches the request to the "partial" strategy: the
    student's reasoning is not what needs correcting, the gap is. The terms
    themselves are named in the prompt, which stays inside the module's
    invariant -- key_terms are a separate bank field from ``correct_answer``,
    ``accepted_variants`` and ``answer_explanation``, none of which are sent.

    Otherwise, a ``matched_bank_id`` naming one of the question's own
    misconceptions switches to the targeted prompt (see ``build_hint_prompt``).
    """
    prompt = build_hint_prompt(
        question, answer, error_type, missing_terms=missing_terms, matched_bank_id=matched_bank_id
    )
    result = client.generate(prompt, max_tokens=MAX_TOKENS)
    if result.get("error") is not None:
        return None
    return str(result.get("text", "")).strip()


def _content_ngrams(text: str, n: int = PARAPHRASE_NGRAM) -> set[tuple[str, ...]]:
    """Runs of ``n`` consecutive content-word stems (stopwords dropped, crude_stem applied)."""
    stems = [crude_stem(t) for t in tokenize(text)]
    return {tuple(stems[i : i + n]) for i in range(len(stems) - n + 1)}


def shared_answer_ngrams(hint: str, question: Any, n: int = PARAPHRASE_NGRAM) -> set[tuple[str, ...]]:
    """Content-word n-grams the hint shares with natural_correct_example, correct_answer or a variant."""
    refs = [getattr(question, "natural_correct_example", "") or "", question.correct_answer, *question.accepted_variants]
    answer_grams: set[tuple[str, ...]] = set()
    for ref in refs:
        answer_grams |= _content_ngrams(ref, n)
    return _content_ngrams(hint, n) & answer_grams


def key_term_forms(term: str) -> list[str]:
    """The term, its synonyms.expand_terms() entries, then its KEY_TERM_FORMS, deduplicated."""
    return list(dict.fromkeys([*expand_terms(term), *KEY_TERM_FORMS.get(term.lower(), [])]))


def named_key_terms(hint: str, question: Any) -> list[str]:
    """``term~form`` for every key-term form the hint contains: as a phrase, or
    (single words) by crude_stem, so "divides" matches the form "divide"."""
    words = normalize(hint).split()
    padded = f" {' '.join(words)} "
    stems = {crude_stem(w) for w in words}
    hits = []
    for term in getattr(question, "key_terms", ()):
        for form in key_term_forms(term):
            f = normalize(form)
            if f and (f" {f} " in padded or (" " not in f and crude_stem(f) in stems)):
                hits.append(f"{term}~{form}")
    return hits


def validate_hint(hint: str, question: Any, *, targeted: bool = False) -> tuple[bool, str]:
    """Pure-Python gate: (True, "ok") or (False, reason). Never calls the model.

    ``targeted`` (a matched-misconception prompt) adds rule 3c.
    """
    stripped = hint.strip()
    if not stripped:
        return False, "empty hint"

    # normalize() lowercases and strips punctuation, so "Semi-conservative"
    # is still caught when the hint writes it as "semi conservative".
    hint_norm = normalize(stripped)
    if normalize(question.correct_answer) in hint_norm:
        return False, "contains correct answer"
    for variant in question.accepted_variants:
        if normalize(variant) in hint_norm:
            return False, f"contains accepted variant: {variant!r}"
    # 3a: the everyday-words correct answer, verbatim.
    natural = normalize(getattr(question, "natural_correct_example", "") or "")
    if natural and natural in hint_norm:
        return False, "contains natural_correct_example"
    # 3b: a paraphrase that keeps a run of the answer's content words.
    shared = shared_answer_ngrams(stripped, question)
    if shared:
        return False, f"paraphrases the answer: {' '.join(sorted(shared)[0])!r}"
    # 3c: on the targeted path a key term is the correct mechanism, e.g. offered
    # as the other half of an either/or. The untargeted prompts never leaked it.
    if targeted and named_key_terms(stripped, question):
        return False, "names a key term on the targeted path"

    words = len(stripped.split())
    if words > MAX_HINT_WORDS:
        return False, f"too long: {words} words > {MAX_HINT_WORDS}"
    if not stripped.endswith("?"):
        return False, "does not end with '?'"
    if stripped.endswith("??"):
        return False, "ends with '??'"
    return True, "ok"


def hint_pipeline(
    question: Any,
    answer: str,
    error_type: str,
    client: Any,
    *,
    missing_terms: Optional[Sequence[str]] = None,
    matched_bank_id: Optional[str] = None,
) -> dict:
    """Generate, validate, retry once, else fall back to a template.

    With ``missing_terms`` the bar is higher: a hint that clears the validator
    but names none of the omitted ideas is rejected too, because a partial
    answer needs pointing at the gap. When the model will not do that, the
    deterministic ``PARTIAL_HINT_TEMPLATE`` does -- and is itself validated,
    so the partial path can never leak what the model path could not.
    ``matched_bank_id`` (the classifier's stable id) targets the prompt at
    that misconception when it is one of this question's.
    """
    missing = _clean_terms(missing_terms)
    # Same precedence as build_hint_prompt: partial wins over a matched misconception.
    targeted = not missing and resolve_misconception(question, matched_bank_id) is not None
    rejections: list[str] = []
    for _ in range(2):
        hint = generate_hint(
            question, answer, error_type, client, missing_terms=missing, matched_bank_id=matched_bank_id
        )
        if hint is None:
            # Counted, so a fallback caused by the client is visible in the log.
            rejections.append("client error")
            break
        if not hint:
            break
        ok, reason = validate_hint(hint, question, targeted=targeted)
        if not ok:
            rejections.append(reason)
            continue
        if missing and not names_a_missing_term(hint, missing):
            rejections.append("names no missing key term")
            continue
        return {"hint": hint, "source": "llm", "rejections": rejections}

    if missing:
        templated = PARTIAL_HINT_TEMPLATE.format(term=missing[0])
        if validate_hint(templated, question)[0]:
            return {"hint": templated, "source": "partial_template", "rejections": rejections}
    return {"hint": question.fallback_hint, "source": "template", "rejections": rejections}
