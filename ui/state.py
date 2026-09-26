"""state.py — session wiring and pure readers over SocraticSession history.

Two halves, deliberately separated:

* Pure functions (``question_ids_for_clusters``, ``build_session``, and the
  ``history`` readers) take everything they need as arguments and never touch
  ``st.session_state``. Tests call these directly.
* Streamlit-facing helpers (``init_state``, ``rebuild_session``, the cached
  loaders) own the ``sc_``-prefixed keys.

Only the *filter* tier is ever scheduled. That mirrors the state machine's own
default (``question_ids=None`` walks filters only): Socratic questions are
escalation targets, injected mid-session by ``escalation.choose_escalation``,
not items a student walks through in order.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import streamlit as st

from socratic_core.classifier_llm import MATCHED_NONE, make_classifier_fn, make_verify_fn
from socratic_core.cloud_client import GroqClient, groq_available
from socratic_core.hint_pipeline import hint_pipeline
from socratic_core.local_client import LocalCPUClient
from socratic_core.mastery import INITIAL_MASTERY
from socratic_core.mock_client import MockInferenceClient
from socratic_core.question_bank import QuestionBank, load_question_bank
from socratic_core.state_machine import SocraticSession

BACKEND_MOCK = "Mock (instant)"
BACKEND_LOCAL = "Local CPU (~7s)"
# Off-device: demo latency only, never the default (CORRECTIONS.md #8).
BACKEND_GROQ = "Groq cloud (off-device)"
BACKEND_LABELS: tuple[str, ...] = (BACKEND_MOCK, BACKEND_LOCAL, BACKEND_GROQ)

ANSWER_EVENT = "answer"
# The only entries a turn logs before its own ``answer`` entry.
_PRE_ANSWER_EVENTS = frozenset({"llm_classify", "classifier_error"})

# error_source -> where in the pipeline the verdict was decided.
_RESOLVED_AT: dict[str, str] = {
    "behavioural": "layer 1 · behavioural",
    "key_terms": "layer 2 · key terms",
    "key_terms_verified": "layer 2 · key terms (verified)",
    "llm": "layer 3 · LLM classifier",
    "llm_groq_fallback": "layer 3 · LLM classifier (Groq fallback)",
}
_RESOLVED_AT_BANK = "bank exact match"
# error_source values meaning the LLM classifier decided the turn.
_LLM_SOURCES = frozenset({"llm", "llm_groq_fallback"})

# Turn-table display.
_DASH = "—"
_NO_MATCH = "no match"
_HINT_PREVIEW_CHARS = 40


# -- bank / clusters ----------------------------------------------------------


def all_clusters(bank: QuestionBank) -> list[str]:
    """Every cluster in the bank, in bank order."""
    seen: dict[str, None] = {}
    for qid in bank.ids():
        seen.setdefault(bank.get(qid).cluster, None)
    return list(seen)


def question_ids_for_clusters(bank: QuestionBank, clusters: Iterable[str]) -> list[str]:
    """Filter-tier question ids belonging to ``clusters``, in bank order."""
    wanted = set(clusters)
    ids: list[str] = []
    for qid in bank.ids():
        q = bank.get(qid)
        if q.tier == "filter" and q.cluster in wanted:
            ids.append(qid)
    return ids


# -- backends -----------------------------------------------------------------


def make_client(backend_label: str, bank: Optional[QuestionBank] = None) -> Any:
    """The generate()-shaped client for a sidebar backend label.

    ``InferenceClient`` (NPU) is deliberately absent: it only runs on ARM64.
    ``GroqClient`` raises ``GroqConfigError`` without a key; the sidebar
    checks ``groq_available()`` before offering the switch. With ``bank``,
    the mock classifies that bank's partial demo presets as partial.

    An unknown label raises ``ValueError``: falling back to the mock would
    silently grade a "local" session with canned answers (a label persisted
    in session state from before a rename did exactly that).
    """
    if backend_label == BACKEND_LOCAL:
        return LocalCPUClient()
    if backend_label == BACKEND_GROQ:
        return GroqClient()
    if backend_label == BACKEND_MOCK:
        if bank is None:
            return MockInferenceClient()
        from .student_view import mock_label_overrides  # deferred: student_view imports this module

        return MockInferenceClient(label_overrides=mock_label_overrides(bank))
    raise ValueError(f"unknown backend label {backend_label!r}; expected one of {BACKEND_LABELS}")


def cloud_fallback_fn(enabled: bool, backend_label: str) -> Optional[Any]:
    """Groq as the low-confidence classifier fallback, or None.

    Only behind the local backend, only when the sidebar opts in, and only
    with a key configured. Anything else means no fallback.
    """
    if not enabled or backend_label != BACKEND_LOCAL or not groq_available():
        return None
    return make_classifier_fn(get_client(BACKEND_GROQ))


def set_cloud_fallback(enabled: bool) -> None:
    """Sidebar callback: attach/detach the fallback on the live session."""
    st.session_state.sc_session.fallback_classifier_fn = cloud_fallback_fn(
        enabled, st.session_state.sc_backend
    )


def build_session(
    bank: QuestionBank,
    clusters: Iterable[str],
    backend_label: str = BACKEND_MOCK,
    client: Optional[Any] = None,
) -> SocraticSession:
    """A fresh session over ``clusters``. Pass ``client`` to bypass the cache."""
    if client is None:
        client = get_client(backend_label)
    return SocraticSession(
        bank,
        question_ids=question_ids_for_clusters(bank, clusters),
        classifier_fn=make_classifier_fn(client),
        verify_fn=make_verify_fn(client),
        # The whole hint_pipeline dict: the session logs its rejections.
        hint_fn=lambda q, a, e, *, missing_terms=None, matched_bank_id=None: hint_pipeline(
            q, a, e, client, missing_terms=missing_terms, matched_bank_id=matched_bank_id
        ),
        backend=backend_label,
    )


# -- history readers ----------------------------------------------------------


def last_answer_event(history: Sequence[dict]) -> Optional[dict]:
    for entry in reversed(history):
        if entry.get("event") == ANSWER_EVENT:
            return entry
    return None


def last_turn_slice(history: Sequence[dict]) -> list[dict]:
    """Every entry belonging to the most recent turn.

    A turn straddles its own ``answer`` entry. Classification is the only
    thing logged *before* it; ``hint``, ``escalation_choice``, ``topic_gap``
    and ``mastery_snapshot`` all follow. So the slice starts at the classifier
    run immediately preceding the last answer -- not at the previous answer,
    which would drag that turn's trailing entries in with it.
    """
    start: Optional[int] = None
    for i, entry in enumerate(history):
        if entry.get("event") == ANSWER_EVENT:
            start = i
    if start is None:
        return []
    while start > 0 and history[start - 1].get("event") in _PRE_ANSWER_EVENTS:
        start -= 1
    return list(history[start:])


def find_event(entries: Sequence[dict], event: str) -> Optional[dict]:
    for entry in entries:
        if entry.get("event") == event:
            return entry
    return None


def resolved_at(error_source: Optional[str]) -> str:
    if not error_source:
        return _RESOLVED_AT_BANK
    return _RESOLVED_AT.get(error_source, error_source)


def llm_called(error_source: Optional[str]) -> bool:
    """Whether the LLM classifier (local or the Groq fallback) decided the turn."""
    return error_source in _LLM_SOURCES


def matched_label(matched_bank_id: Optional[str]) -> str:
    """Turn-table text: "—" (LLM not invoked), "no match", or the example ID."""
    if matched_bank_id is None:
        return _DASH
    if matched_bank_id == MATCHED_NONE:
        return _NO_MATCH
    return matched_bank_id


def hint_preview(hint_text: Optional[str]) -> str:
    """First ``_HINT_PREVIEW_CHARS`` characters, with "…" only if cut."""
    text = hint_text or ""
    if len(text) <= _HINT_PREVIEW_CHARS:
        return text
    return text[:_HINT_PREVIEW_CHARS] + "…"


def mastery_snapshots(history: Sequence[dict]) -> list[dict]:
    return [e["mastery"] for e in history if isinstance(e.get("mastery"), dict)]


def mastery_delta(
    history: Sequence[dict], cluster: str
) -> tuple[Optional[float], Optional[float]]:
    """``(current, delta)`` for ``cluster``, or ``(None, None)`` if untouched.

    An absent cluster in the previous snapshot means it was being scored for
    the first time, so the baseline is ``INITIAL_MASTERY``.
    """
    snapshots = mastery_snapshots(history)
    if not snapshots:
        return None, None
    current = snapshots[-1].get(cluster)
    if current is None:
        return None, None
    previous = (
        snapshots[-2].get(cluster, INITIAL_MASTERY)
        if len(snapshots) >= 2
        else INITIAL_MASTERY
    )
    return current, current - previous


def answer_rows(history: Sequence[dict]) -> list[dict]:
    """One row per answered turn. ``elapsed_ms`` is turn entry to verdict."""
    rows: list[dict] = []
    for entry in history:
        if entry.get("event") != ANSWER_EVENT:
            continue
        rows.append(
            {
                "turn": len(rows) + 1,
                "backend": entry.get("backend") or _DASH,
                "question_id": entry.get("question_id"),
                "tier": entry.get("tier"),
                "attempt_number": entry.get("attempt_number"),
                "verdict": entry.get("verdict"),
                "error_type": entry.get("error_type"),
                "error_source": entry.get("error_source"),
                "matched_bank_id": matched_label(entry.get("matched_bank_id")),
                "elapsed_ms": entry.get("elapsed_ms"),
                "hint_text": hint_preview(entry.get("hint_text")),
            }
        )
    return rows


# -- streamlit session state --------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_bank() -> QuestionBank:
    return load_question_bank()


@st.cache_resource(show_spinner="Loading backend…")
def get_client(backend_label: str) -> Any:
    return make_client(backend_label, get_bank())


def init_state() -> None:
    """Populate the ``sc_`` keys on first render. Idempotent."""
    if "sc_session" in st.session_state:
        return
    bank = get_bank()
    st.session_state.sc_backend = BACKEND_MOCK
    st.session_state.sc_clusters = all_clusters(bank)
    st.session_state.sc_answer_draft = ""
    st.session_state.sc_last_result = None
    st.session_state.sc_turn_count = 0
    st.session_state.sc_session = build_session(
        bank, st.session_state.sc_clusters, st.session_state.sc_backend
    )


def rebuild_session() -> None:
    """Re-instantiate from the current backend/cluster selection.

    The state machine has no reset; a new instance *is* the reset. Safe to call
    while the sidebar renders, because the answer box has not been instantiated
    yet on that pass.
    """
    st.session_state.sc_session = build_session(
        get_bank(), st.session_state.sc_clusters, st.session_state.sc_backend
    )
    set_cloud_fallback(bool(st.session_state.get("sc_cloud_fallback", False)))
    st.session_state.sc_last_result = None
    st.session_state.sc_turn_count = 0
    st.session_state.sc_answer_draft = ""
