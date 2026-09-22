"""
state_machine.py — the Socratic dialogue loop.

Role in the architecture
------------------------
Owns all control flow. Questions come in two tiers.

A *filter* is a one-fact probe: the answer is matched exactly (after
normalisation) against the bank and nothing else runs. Right -> advance.
Wrong -> the cluster's mastery drops and, if it is now below
``mastery.ESCALATION_THRESHOLD``, ``escalation.choose_escalation`` picks a
Socratic question to inject next; otherwise a ``topic_gap`` is logged and
the session advances. A filter never produces an error label of any kind.

A *socratic* question runs the full pipeline. For every student answer it
first checks correctness against the bank, then walks the classification
layers in a fixed order and stops at the first one that decides:

    1. behavioural: give-up keyword, word_count < min_words, or an off-topic
       answer carrying a disengagement token -> low_effort (0 ms).
       Off-topic on its own does not decide; it falls through to layer 2.
    2. key_terms (Case A/B/C, ``_classify_by_key_terms``):
       all of the question's key_terms present (allowing synonyms,
       ``synonyms.expand_terms``)   -> correct           (0 ms)
       some present, some missing   -> partial            (0 ms)
       none present                 -> continue to layer 3
    3. LLM classifier                -> correct / wording / logic (NPU),
                                        mapped to a verdict by ``LLM_VERDICTS``

then hint generation (LLM + pure-Python validator). The bank's misconceptions
are never looked up as a live classification stage; they exist only as
few-shot context for the LLM classifier's own prompt (``classifier_llm.py``).
``bypass_bank_lookup`` is kept on ``SessionState`` for log-shape compatibility
but is a no-op now that there is no bank lookup to bypass.

After ``max_attempts`` wrong answers on one question the session escalates:
it reveals a minimal two-line answer taken from the bank (never from the
model), marks the question as stuck, and advances.

Per-cluster mastery (``mastery.py``) is updated after every filter answer,
on every correct, partial or wrong Socratic answer (never on low_effort),
and a snapshot is logged on every turn.

The LLM-facing steps are injected as plain callables so this module can be
exercised with no model at all:

    classifier_fn(question, answer)             -> Mapping with
                                                   error_type / confidence / reasoning
                                                   (error_type: correct, wording_error
                                                   or logic_error; optional raw_output)
    hint_fn(question, answer, error_type)       -> hint string

The classifier's label is turned into a verdict only through ``LLM_VERDICTS``:
``correct`` scores and advances exactly like a key_terms Case A answer (no
hint); ``wording_error`` and ``logic_error`` are both a wrong verdict, the
label surviving as ``error_type``. Any other label fails closed to
``logic_error``.

If either is ``None`` a conservative offline default is used (``logic_error``
and the bank's ``fallback_hint``). ``run_session.py`` wires in the real
``classifier_llm`` and ``hint_pipeline`` implementations.

Persistence: one JSON file per session in ``sessions/`` (or
``SOCRATIC_SESSIONS_DIR``), rewritten after every turn so a crash mid-session
loses at most the current turn.
"""

from __future__ import annotations

import inspect
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from . import classifier_behavioral
from .disengagement import flag_disengagement
from .escalation import choose_escalation
from .mastery import should_escalate, update_mastery
from .noise import NOISE_TOLERANCE_PREFIX
from .question_bank import (
    ErrorType,
    Question,
    QuestionBank,
    normalize,
)
from .synonyms import expand_terms

ENV_SESSIONS_DIR = "SOCRATIC_SESSIONS_DIR"
DEFAULT_SESSIONS_DIRNAME = "sessions"
DEFAULT_MAX_ATTEMPTS = 3

# LLM classifier label -> (verdict, kind). The only place an LLM label becomes
# a verdict. wording_error is an error_type, not a verdict: it scores wrong.
LLM_VERDICTS: dict[str, tuple[str, str]] = {
    "correct": ("correct", "socratic_correct"),
    "logic_error": ("wrong", "socratic_wrong"),
    "wording_error": ("wrong", "socratic_wrong"),
}
LLM_FALLBACK_LABEL = "logic_error"

ClassifierFn = Callable[[Question, str], Mapping[str, Any]]
HintFn = Callable[[Question, str, ErrorType], str]


# -- data ---------------------------------------------------------------------


@dataclass
class Classification:
    """Normalised result of whichever layer decided the error type."""

    error_type: ErrorType
    source: str  # "bank" | "llm" | "key_terms" | "behavioural" (see question_bank error_source enum)
    confidence: float = 1.0
    reasoning: str = ""


@dataclass
class TurnResult:
    """What the caller (CLI / UI) should show after one student answer."""

    kind: str  # "correct" | "hint" | "escalated"
    question_id: str
    attempt: int
    message: str  # hint text, or the two-line reveal on escalation
    error_type: Optional[ErrorType] = None
    error_source: Optional[str] = None
    session_finished: bool = False
    next_question: Optional[Question] = None


@dataclass
class SessionState:
    session_id: str
    question_ids: list[str]
    question_index: int = 0
    question_id: Optional[str] = None
    attempt_count: int = 0
    last_error_type: Optional[ErrorType] = None
    history: list[dict[str, Any]] = field(default_factory=list)
    stuck_question_ids: list[str] = field(default_factory=list)
    solved_question_ids: list[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: Optional[str] = None
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    bypass_bank_lookup: bool = False
    mastery: dict[str, float] = field(default_factory=dict)

    @property
    def finished(self) -> bool:
        return self.question_index >= len(self.question_ids)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _new_session_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def default_sessions_dir() -> Path:
    env = os.environ.get(ENV_SESSIONS_DIR)
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / DEFAULT_SESSIONS_DIRNAME


# -- offline defaults (used when no LLM callables are injected) ----------------


def default_classifier(question: Question, answer: str) -> Mapping[str, Any]:
    """Conservative stand-in for the LLM classifier: always ``logic_error``."""
    return {
        "error_type": "logic_error",
        "confidence": 0.0,
        "reasoning": "no LLM classifier configured",
    }


def default_hint(question: Question, answer: str, error_type: ErrorType) -> str:
    """Stand-in for the hint pipeline: the bank's hand-written fallback."""
    return question.fallback_hint


# -- the machine --------------------------------------------------------------


def _hint_fn_accepts_missing_terms(fn: HintFn) -> bool:
    """Whether ``fn`` can be handed the partial path's missing key terms.

    ``hint_fn`` is a public injection point with a three-argument contract, so
    the extra keyword is offered only to callables that declare it. Anything
    older keeps being called exactly as before.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return "missing_terms" in params


class SocraticSession:
    """Step-driven dialogue loop. Call ``current_question()`` then
    ``submit_answer()`` repeatedly until ``state.finished``."""

    def __init__(
        self,
        bank: QuestionBank,
        question_ids: Optional[Sequence[str]] = None,
        *,
        classifier_fn: Optional[ClassifierFn] = None,
        hint_fn: Optional[HintFn] = None,
        sessions_dir: str | os.PathLike[str] | None = None,
        session_id: Optional[str] = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        autosave: bool = True,
        bypass_bank_lookup: bool = False,
    ) -> None:
        # A session walks the filters; Socratic entries are reached only by
        # escalation. A bank with no filters is walked in full (legacy shape).
        if question_ids is not None:
            ids = list(question_ids)
        else:
            ids = [q.id for q in bank if q.tier == "filter"] or bank.ids()
        for qid in ids:
            if qid not in bank:
                raise KeyError(f"unknown question id: {qid!r}")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

        self.bank = bank
        self.classifier_fn: ClassifierFn = classifier_fn or default_classifier
        self.hint_fn: HintFn = hint_fn or default_hint
        self._hint_fn_takes_missing_terms = _hint_fn_accepts_missing_terms(self.hint_fn)
        self.sessions_dir = Path(sessions_dir) if sessions_dir else default_sessions_dir()
        self.autosave = autosave
        # Evaluation switch: skip the misconception layer so every non-low-effort
        # wrong answer reaches the classifier.
        self.bypass_bank_lookup = bypass_bank_lookup
        self.state = SessionState(
            session_id=session_id or _new_session_id(),
            question_ids=ids,
            started_at=_now_iso(),
            max_attempts=max_attempts,
            bypass_bank_lookup=bypass_bank_lookup,
        )
        self._present_current()

    # -- public API ----------------------------------------------------------

    @property
    def log_path(self) -> Path:
        return self.sessions_dir / f"{self.state.session_id}.json"

    def current_question(self) -> Optional[Question]:
        if self.state.finished:
            return None
        return self.bank.get(self.state.question_ids[self.state.question_index])

    def submit_answer(self, answer: str) -> TurnResult:
        question = self.current_question()
        if question is None:
            raise RuntimeError("session is finished; no question to answer")

        st = self.state
        answer = answer.strip()

        if question.tier == "filter":
            # A filter is a probe, not a lesson: exact match after normalisation
            # and nothing else. No behavioural rules, no min_words, no bank
            # lookup, no LLM, no hint. Empty, "idk", gibberish and wrong all
            # take the same path, so a filter can never yield low_effort.
            st.attempt_count += 1
            correct = question.is_correct(answer)
            disengaged, disengagement_tokens = flag_disengagement(answer)
            self._log(
                "answer",
                question,
                tier=question.tier,
                answer=answer,
                attempt=st.attempt_count,
                correct=correct,
                verdict="correct" if correct else "wrong",
                kind="correct" if correct else "wrong",
                disengagement_flag=disengaged,
                disengagement_tokens=disengagement_tokens,
            )
            update_mastery(st.mastery, question.cluster, question.tier, correct)
            if correct:
                st.solved_question_ids.append(question.id)
                self._log("correct", question, attempt=st.attempt_count)
                return self._advance("correct", question, "Correct.")
            if should_escalate(st.mastery, question.cluster):
                candidate = choose_escalation(question, self.bank, st.mastery)
                if candidate is not None:
                    st.question_ids.insert(st.question_index + 1, candidate.id)
                    self._log(
                        "escalation_choice",
                        question,
                        from_filter_id=question.id,
                        to_question_id=candidate.id,
                        reason="cluster_match" if candidate.cluster == question.cluster else "global_fallback",
                    )
                    return self._advance("filter_escalated", question, "Not quite. Let's look at this more closely.")
                self._log("topic_gap", question, cluster=question.cluster, reason="no_socratic_candidate")
            else:
                self._log("topic_gap", question, cluster=question.cluster, reason="mastery_above_threshold")
            return self._advance("filter_missed", question, "Not quite. Let's move on.")

        disengaged, disengagement_tokens = flag_disengagement(answer)

        if question.is_correct(answer):
            st.attempt_count += 1
            self._log(
                "answer",
                question,
                tier=question.tier,
                answer=answer,
                attempt=st.attempt_count,
                correct=True,
                verdict="correct",
                kind="correct",
                key_terms_matched=list(question.key_terms),
                key_terms_missing=[],
                disengagement_flag=disengaged,
                disengagement_tokens=disengagement_tokens,
            )
            st.solved_question_ids.append(question.id)
            update_mastery(st.mastery, question.cluster, question.tier, True)
            self._log("correct", question, attempt=st.attempt_count)
            return self._advance("correct", question, "Correct.")

        st.attempt_count += 1

        # Layer 1: behavioural (give-up keyword / too short / off-topic paired
        # with a disengagement token). Off-topic alone falls through.
        reason = classifier_behavioral.explain(answer, question)
        kt_verdict: Optional[str] = None
        matched: list[str] = []
        missing: list[str] = list(question.key_terms)
        if reason is None:
            # Layer 2: key_terms Case A/B/C, before the LLM ever runs.
            kt_verdict, matched, missing = self._classify_by_key_terms(question, answer)

        if kt_verdict == "correct":
            return self._socratic_correct(
                question,
                answer,
                error_source="key_terms",
                matched=matched,
                missing=missing,
                disengaged=disengaged,
                disengagement_tokens=disengagement_tokens,
            )

        if kt_verdict == "partial":
            # error_type stays logic_error: it is the label that goes to the
            # log and to TurnResult, and "partial" is a verdict, not one of
            # VALID_ERROR_TYPES. The hint layer is steered by missing_terms
            # instead, which selects HINT_STRATEGY["partial"] downstream.
            cls = Classification("logic_error", "key_terms", 1.0, "partial key-term match")
            update_mastery(st.mastery, question.cluster, "socratic", "partial")
            st.last_error_type = cls.error_type
            self._log(
                "answer",
                question,
                tier=question.tier,
                answer=answer,
                attempt=st.attempt_count,
                correct=False,
                verdict="partial",
                kind="socratic_partial",
                error_type=cls.error_type,
                error_source=cls.source,
                key_terms_matched=matched,
                key_terms_missing=missing,
                disengagement_flag=disengaged,
                disengagement_tokens=disengagement_tokens,
            )
            hint = self._hint(question, answer, cls.error_type, missing_terms=missing)
            self._log("hint", question, attempt=st.attempt_count, error_type=cls.error_type, hint=hint)
            self._log("mastery_snapshot", question, mastery=dict(st.mastery))
            self._maybe_save()
            return TurnResult(
                kind="hint",
                question_id=question.id,
                attempt=st.attempt_count,
                message=hint,
                error_type=cls.error_type,
                error_source=cls.source,
                session_finished=False,
                next_question=question,
            )

        if reason is not None:
            cls = Classification("low_effort", "behavioural", 1.0, reason)
            verdict, kind = "low_effort", "low_effort"
        else:
            # Layer 3: zero key_terms matched (Case C) -> the LLM classifier,
            # whose label becomes a verdict only through LLM_VERDICTS.
            cls, verdict, kind = self._classify_llm(question, answer)
            if verdict == "correct":
                return self._socratic_correct(
                    question,
                    answer,
                    error_source=cls.source,
                    matched=matched,
                    missing=missing,
                    disengaged=disengaged,
                    disengagement_tokens=disengagement_tokens,
                    kind=kind,
                    turn_error_source=cls.source,
                    confidence=cls.confidence,
                    reasoning=cls.reasoning,
                )

        st.last_error_type = cls.error_type
        self._log(
            "answer",
            question,
            tier=question.tier,
            answer=answer,
            attempt=st.attempt_count,
            correct=(verdict == "correct"),
            verdict=verdict,
            kind=kind,
            error_type=cls.error_type,
            error_source=cls.source,
            confidence=cls.confidence,
            reasoning=cls.reasoning,
            key_terms_matched=matched,
            key_terms_missing=missing,
            disengagement_flag=disengaged,
            disengagement_tokens=disengagement_tokens,
        )
        # Every wrong attempt costs mastery immediately; low_effort does not.
        if verdict == "wrong":
            update_mastery(st.mastery, question.cluster, question.tier, False)

        if st.attempt_count >= st.max_attempts:
            reveal = self._reveal_text(question)
            st.stuck_question_ids.append(question.id)
            self._log("escalation", question, attempt=st.attempt_count, reveal=reveal)
            return self._advance("escalated", question, reveal, cls)

        hint = self._hint(question, answer, cls.error_type)
        self._log("hint", question, attempt=st.attempt_count, error_type=cls.error_type, hint=hint)
        self._log("mastery_snapshot", question, mastery=dict(st.mastery))
        self._maybe_save()
        return TurnResult(
            kind="hint",
            question_id=question.id,
            attempt=st.attempt_count,
            message=hint,
            error_type=cls.error_type,
            error_source=cls.source,
            session_finished=False,
            next_question=question,
        )

    def save(self) -> Path:
        """Write the session log (atomically) and return its path."""
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.log_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self.state.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp, self.log_path)
        return self.log_path

    # -- layers --------------------------------------------------------------

    @staticmethod
    def _classify_by_key_terms(question: Question, answer: str) -> tuple[Optional[str], list[str], list[str]]:
        """Case A/B/C over ``question.key_terms`` (synonym-expanded), 0 ms.

        Returns ``(verdict, matched, missing)``:
            "correct" -> every key term present (directly or via a synonym)
            "partial" -> at least one present, at least one missing
            None      -> none present; caller falls through to the LLM
        """
        haystack = f" {normalize(answer)} "
        matched: list[str] = []
        missing: list[str] = []
        for term in question.key_terms:
            if any(f" {normalize(variant)} " in haystack for variant in expand_terms(term)):
                matched.append(term)
            else:
                missing.append(term)
        if not missing:
            return "correct", matched, missing
        if matched:
            return "partial", matched, missing
        return None, matched, missing

    def _classify_llm(self, question: Question, answer: str) -> tuple[Classification, str, str]:
        """The LLM classifier, reached only when key_terms matched nothing (Case C).

        Returns ``(classification, verdict, kind)``, the last two looked up in
        ``LLM_VERDICTS``. An unknown label, or any failure, degrades to
        logic_error / wrong rather than breaking the session; it never becomes
        correct. The noise-tolerance prefix travels inside the ``answer`` text
        handed to ``classifier_fn`` because the prompt itself is built inside
        classifier_llm.py.
        """
        t0 = time.perf_counter()
        try:
            raw = self.classifier_fn(question, f"{NOISE_TOLERANCE_PREFIX} {answer}")
        except Exception as e:  # noqa: BLE001 - deliberate: never crash the loop
            self._log("classifier_error", question, error=repr(e))
            verdict, kind = LLM_VERDICTS[LLM_FALLBACK_LABEL]
            return Classification(LLM_FALLBACK_LABEL, "llm", 0.0, f"classifier failed: {e!r}"), verdict, kind
        elapsed_ms = (time.perf_counter() - t0) * 1000

        llm_label = str(raw.get("error_type", ""))
        label = llm_label if llm_label in LLM_VERDICTS else LLM_FALLBACK_LABEL
        verdict, kind = LLM_VERDICTS[label]
        cls = Classification(
            label,  # type: ignore[arg-type]
            "llm",
            float(raw.get("confidence", 0.0)),
            str(raw.get("reasoning", "")),
        )
        extra: dict[str, Any] = {}
        if "raw_output" in raw:
            extra["raw_output"] = str(raw["raw_output"])
        elif label != llm_label:
            extra["raw_output"] = repr(dict(raw))
        self._log(
            "llm_classify",
            question,
            elapsed_ms=round(elapsed_ms, 1),
            **asdict(cls),
            llm_label=llm_label,
            verdict=verdict,
            kind=kind,
            **extra,
        )
        return cls, verdict, kind

    def _socratic_correct(
        self,
        question: Question,
        answer: str,
        *,
        error_source: str,
        matched: list[str],
        missing: list[str],
        disengaged: bool,
        disengagement_tokens: list[str],
        kind: str = "socratic_correct",
        turn_error_source: Optional[str] = None,
        **log_extra: Any,
    ) -> TurnResult:
        """Score a Socratic answer correct and advance: key_terms Case A, or an
        LLM ``correct``. No hint. ``error_source`` goes to the answer log;
        ``turn_error_source`` to the returned TurnResult (None for Case A)."""
        st = self.state
        self._log(
            "answer",
            question,
            tier=question.tier,
            answer=answer,
            attempt=st.attempt_count,
            correct=True,
            verdict="correct",
            kind=kind,
            error_source=error_source,
            key_terms_matched=matched,
            key_terms_missing=missing,
            disengagement_flag=disengaged,
            disengagement_tokens=disengagement_tokens,
            **log_extra,
        )
        st.solved_question_ids.append(question.id)
        update_mastery(st.mastery, question.cluster, question.tier, True)
        self._log("correct", question, attempt=st.attempt_count)
        return self._advance("correct", question, "Correct.", error_source=turn_error_source)

    def _hint(
        self,
        question: Question,
        answer: str,
        error_type: ErrorType,
        missing_terms: Optional[Sequence[str]] = None,
    ) -> str:
        t0 = time.perf_counter()
        try:
            if missing_terms and self._hint_fn_takes_missing_terms:
                hint = self.hint_fn(question, answer, error_type, missing_terms=list(missing_terms))
            else:
                hint = self.hint_fn(question, answer, error_type)
        except Exception as e:  # noqa: BLE001
            self._log("hint_error", question, error=repr(e))
            hint = question.fallback_hint
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if self.hint_fn is not default_hint:
            self._log("llm_hint", question, elapsed_ms=round(elapsed_ms, 1))
        return hint.strip() or question.fallback_hint

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _reveal_text(question: Question) -> str:
        """Minimal two-line answer shown on escalation. Comes from the bank
        only; the model is never asked for it."""
        line1 = f"The answer is: {question.correct_answer}."
        line2 = question.answer_explanation.strip() or (
            f"Re-read the question and note how '{question.correct_answer}' answers it."
        )
        return f"{line1}\n{line2}"

    def _advance(
        self,
        kind: str,
        question: Question,
        message: str,
        cls: Optional[Classification] = None,
        error_source: Optional[str] = None,
    ) -> TurnResult:
        """Move to the next question. ``error_source`` is reported on the
        TurnResult when there is no ``cls`` (an LLM-correct turn: no error_type,
        but the llm decided it)."""
        st = self.state
        attempt = st.attempt_count
        self._log("mastery_snapshot", question, mastery=dict(st.mastery))
        st.question_index += 1
        st.attempt_count = 0
        st.last_error_type = None
        if st.finished:
            st.question_id = None
            st.finished_at = _now_iso()
            self._log("session_finished", None)
        else:
            self._present_current()
        self._maybe_save()
        return TurnResult(
            kind=kind,
            question_id=question.id,
            attempt=attempt,
            message=message,
            error_type=cls.error_type if cls else None,
            error_source=cls.source if cls else error_source,
            session_finished=st.finished,
            next_question=self.current_question(),
        )

    def _present_current(self) -> None:
        q = self.current_question()
        if q is None:
            return
        self.state.question_id = q.id
        self.state.attempt_count = 0
        self.state.last_error_type = None
        self._log("question_presented", q, question_text=q.question_text)

    def _log(self, event: str, question: Optional[Question], **fields: Any) -> None:
        entry: dict[str, Any] = {"ts": _now_iso(), "event": event}
        if question is not None:
            entry["question_id"] = question.id
        entry.update(fields)
        self.state.history.append(entry)

    def _maybe_save(self) -> None:
        if self.autosave:
            self.save()
