"""
classifier_llm.py — layer 4 of error classification (the only NPU-backed one).

Role in the architecture
------------------------
Reached only when the behavioural rules and the bank misconception lookup
have both declined to decide. Asks the model one question — *why* is this
wrong answer wrong — and maps the reply onto the two remaining labels:

    wording_error  right concept, wrong term
    logic_error    right terms, broken reasoning

``low_effort`` is never produced here; the behavioural layer owns it.

The model classifies, it does not teach: its output is a label plus a
one-line justification that goes to the session log. Nothing it says is
shown to the student, which is why the prompt may include the correct
answer (judging "right concept, wrong term" needs it) without any risk of
leaking it.

Fallbacks, in this exact order, always land on ``logic_error`` — the
conservative label whose hint strategy (counter-example) is least likely to
mislead when we are unsure:

    1. client reported an error        -> confidence 0.0, "client error"
    2. no label in the reply           -> confidence 0.0, "unparseable output"
    3. confidence below threshold      -> parsed confidence/reasoning kept for the log
    4. otherwise                       -> the parsed label

The few-shot examples are copied from the bank's own misconceptions. Any
example belonging to the question being classified is dropped, so an
accuracy run with ``bypass_bank_lookup=True`` measures the model, not the
bank echoed back through the prompt.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from .inference_client import build_prompt
from .question_bank import normalize

CONFIDENCE_THRESHOLD = 0.7
FALLBACK_LABEL = "logic_error"
MAX_TOKENS = 96

_LABELS = ("wording_error", "logic_error")

# (question_text, wrong_answer, label, explanation), verbatim from question_bank.json.
# No question contributes more than one example per class, so dropping the
# current question's rows still leaves at least three per class.
_SEED_EXAMPLES: tuple[tuple[str, str, str, str], ...] = (
    (
        "According to cell theory, what is the basic structural and functional unit of all living organisms?",
        "the smallest living thing",
        "wording_error",
        "Right idea, but cell theory names that unit: the cell.",
    ),
    (
        "According to cell theory, where do all new cells come from?",
        "cells make cells",
        "wording_error",
        "Correct idea, but the theory states it precisely: new cells arise from pre-existing cells.",
    ),
    (
        "Describe how the parental strands are distributed in daughter DNA molecules after replication.",
        "Conservative",
        "wording_error",
        "This confuses the process with a similar-sounding alternative.",
    ),
    (
        "What are the three phases of interphase?",
        "Growth, rest, division",
        "wording_error",
        "Right ideas, but the wrong terms for the phases.",
    ),
    (
        "According to cell theory, what is the basic structural and functional unit of all living organisms?",
        "the atom",
        "logic_error",
        "Atoms are the basic unit of matter, not of life. They are not alive.",
    ),
    (
        "According to cell theory, where do all new cells come from?",
        "they form on their own",
        "logic_error",
        "This is spontaneous generation, which experiments disproved. Cells do not arise from non-living matter.",
    ),
    (
        "Which organelle produces ATP?",
        "The cell makes energy in the cytoplasm",
        "logic_error",
        "This answer misses the role of a specialised membrane-bound organelle.",
    ),
    (
        "What are the three phases of interphase?",
        "Prophase, Metaphase, Anaphase",
        "logic_error",
        "This confuses interphase with M phase.",
    ),
    (
        "What is the role of the mutation operator in a genetic algorithm?",
        "It combines two parents to make a child",
        "logic_error",
        "This confuses mutation with crossover.",
    ),
)

SYSTEM_PROMPT = (
    "You are grading a student's wrong answer for a tutor. Do not teach and do not "
    "give the answer. Decide only WHY the answer is wrong, using exactly one label:\n"
    "  wording_error - the student has the right concept but used the wrong term.\n"
    "  logic_error   - the student used the right terms but the reasoning is broken "
    "or names the wrong thing.\n"
    "Reply in exactly this format and nothing else:\n"
    "LABEL: <wording_error or logic_error>\n"
    "CONFIDENCE: <number from 0.0 to 1.0>\n"
    "REASONING: <one sentence>"
)

_LABEL_LINE = re.compile(r"LABEL\s*:\s*(wording_error|logic_error)", re.IGNORECASE)
_LABEL_ANY = re.compile(r"\b(wording_error|logic_error)\b", re.IGNORECASE)
_CONFIDENCE = re.compile(r"CONFIDENCE\s*:\s*([0-9]*\.?[0-9]+)\s*(%?)", re.IGNORECASE)
_REASONING = re.compile(r"REASONING\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


# -- question access (bank dataclass or plain dict) ------------------------------


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


# -- prompt ----------------------------------------------------------------------


def _example_block(question_text: str, wrong_answer: str, label: str, explanation: str) -> str:
    return (
        f"Question: {question_text}\n"
        f"Student answer: {wrong_answer}\n"
        f"LABEL: {label}\n"
        f"CONFIDENCE: 0.95\n"
        f"REASONING: {explanation}"
    )


def few_shot_examples(question: Any) -> list[tuple[str, str, str, str]]:
    """Seed examples with the current question's own rows removed."""
    current = normalize(str(_field(question, "question_text", "")))
    return [ex for ex in _SEED_EXAMPLES if normalize(ex[0]) != current]


def build_classifier_prompt(answer: str, question: Any) -> str:
    examples = "\n\n".join(_example_block(*ex) for ex in few_shot_examples(question))
    case = (
        f"Question: {_field(question, 'question_text', '')}\n"
        f"Correct answer (for your judgement only, never repeat it): "
        f"{_field(question, 'correct_answer', '')}\n"
        f"Student answer: {answer.strip()}"
    )
    user = f"Examples:\n\n{examples}\n\nNow classify this one.\n\n{case}"
    return build_prompt(user, system=SYSTEM_PROMPT)


# -- parsing ---------------------------------------------------------------------


def parse_classifier_output(text: str) -> Optional[dict[str, Any]]:
    """Extract label, confidence and reasoning; ``None`` if no label is present."""
    m = _LABEL_LINE.search(text) or _LABEL_ANY.search(text)
    if m is None:
        return None
    label = m.group(1).lower()

    confidence = 0.0
    c = _CONFIDENCE.search(text)
    if c is not None:
        confidence = float(c.group(1))
        if c.group(2) or confidence > 1.0:
            confidence /= 100.0
        confidence = max(0.0, min(1.0, confidence))

    r = _REASONING.search(text)
    reasoning = r.group(1).strip() if r else ""
    return {"error_type": label, "confidence": confidence, "reasoning": reasoning}


# -- public API ------------------------------------------------------------------


def classify_llm(answer: str, question: Any, client: Any) -> dict:
    """Classify a wrong answer as ``wording_error`` or ``logic_error``.

    ``client`` is anything with ``.generate(prompt, max_tokens) -> dict`` in
    the ``InferenceClient`` shape; ``MockInferenceClient`` works offline.
    """
    result = client.generate(build_classifier_prompt(answer, question), max_tokens=MAX_TOKENS)

    if result.get("error") is not None:
        return {"error_type": FALLBACK_LABEL, "confidence": 0.0, "reasoning": "client error"}

    parsed = parse_classifier_output(str(result.get("text", "")))
    if parsed is None:
        return {"error_type": FALLBACK_LABEL, "confidence": 0.0, "reasoning": "unparseable output"}

    if parsed["confidence"] < CONFIDENCE_THRESHOLD:
        return {**parsed, "error_type": FALLBACK_LABEL}

    return parsed


def make_classifier_fn(client: Any):
    """Adapter to the state machine's ``classifier_fn(question, answer)`` slot."""
    return lambda question, answer: classify_llm(answer, question, client)
