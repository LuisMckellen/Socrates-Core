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

from socratic_core.classifier_llm import make_classifier_fn
from socratic_core.hint_pipeline import hint_pipeline
from socratic_core.local_client import LocalCPUClient
from socratic_core.mastery import INITIAL_MASTERY
from socratic_core.mock_client import MockInferenceClient
from socratic_core.question_bank import QuestionBank, load_question_bank
from socratic_core.state_machine import SocraticSession

BACKEND_MOCK = "Mock (instant)"
BACKEND_LOCAL = "Local CPU (~2s)"
BACKEND_LABELS: tuple[str, ...] = (BACKEND_MOCK, BACKEND_LOCAL)

ANSWER_EVENT = "answer"
# The only entries a turn logs before its own ``answer`` entry.
_PRE_ANSWER_EVENTS = frozenset({"llm_classify", "classifier_error"})

# error_source -> where in the pipeline the verdict was decided.
_RESOLVED_AT: dict[str, str] = {
    "behavioural": "layer 1 · behavioural",
    "key_terms": "layer 2 · key terms",
    "llm": "layer 3 · LLM classifier",
}
_RESOLVED_AT_BANK = "bank exact match"


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


def make_client(backend_label: str) -> Any:
    """The generate()-shaped client for a sidebar backend label.

    ``InferenceClient`` (NPU) is deliberately absent: it only runs on ARM64.
    """
    if backend_label == BACKEND_LOCAL:
        return LocalCPUClient()
    return MockInferenceClient()


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
        hint_fn=lambda q, a, e: hint_pipeline(q, a, e, client)["hint"],
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
    """One row per answered turn, with the LLM latency that preceded it."""
    rows: list[dict] = []
    pending_ms: Optional[float] = None
    for entry in history:
        event = entry.get("event")
        if event == "llm_classify":
            pending_ms = entry.get("elapsed_ms")
        elif event == ANSWER_EVENT:
            rows.append(
                {
                    "turn": len(rows) + 1,
                    "question_id": entry.get("question_id"),
                    "tier": entry.get("tier"),
                    "verdict": entry.get("verdict"),
                    "error_type": entry.get("error_type"),
                    "error_source": entry.get("error_source"),
                    "elapsed_ms": pending_ms,
                }
            )
            pending_ms = None
    return rows


# -- streamlit session state --------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_bank() -> QuestionBank:
    return load_question_bank()


@st.cache_resource(show_spinner="Loading backend…")
def get_client(backend_label: str) -> Any:
    return make_client(backend_label)


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
    st.session_state.sc_last_result = None
    st.session_state.sc_turn_count = 0
    st.session_state.sc_answer_draft = ""
