"""student_view.py — the question, the answer box and the feedback card.

Everything that mutates the session runs in an ``on_click`` callback. Streamlit
runs callbacks *before* the script re-executes, which is the only point at
which ``sc_answer_draft`` may be written: once the text area has been
instantiated on a given pass, assigning to its key raises.
"""

from __future__ import annotations

import re
from html import escape
from typing import Iterable, Optional

import streamlit as st

from socratic_core.question_bank import Question
from socratic_core.state_machine import SocraticSession, TurnResult

from . import state

_BADGE = (
    "<span style='background:{bg};color:{fg};padding:3px 11px;border-radius:999px;"
    "font-size:0.75rem;font-weight:600;letter-spacing:0.03em;margin-right:7px;"
    "font-family:ui-monospace,SFMono-Regular,Menlo,monospace;'>{text}</span>"
)

_VERDICT_COLOUR: dict[str, str] = {
    "correct": "#16A34A",  # green
    "partial": "#D97706",  # amber
    "wrong": "#DC2626",  # red
    "low_effort": "#6B7280",  # grey
}

PRESET_GIVE_UP = "No idea"
_LABEL_LIMIT = 72


def _drop_first_key_term(question: Question) -> Optional[str]:
    """The canonical variant minus its first key term: a *partial* answer.

    Produces a response that still carries the question's other key terms, so
    ``_classify_by_key_terms`` finds a term missing and the answer is left to
    the LLM partial label rather than Case A.

    The removal takes the whole word, not the bare substring: key term "gene"
    appears in the canonical text as "genes", and cutting the substring would
    strand an "s" in the middle of the button label.
    """
    if not question.key_terms or not question.accepted_variants:
        return None
    canonical = question.accepted_variants[0]
    stripped = re.sub(
        rf"\b{re.escape(question.key_terms[0])}\w*\b", "", canonical, count=1
    )
    stripped = re.sub(r"\s+([,;.])", r"\1", stripped)  # close the gap left behind
    stripped = " ".join(stripped.split()).strip(" ,;.")
    if not stripped or stripped == canonical:
        return None
    return stripped


def build_presets(question: Question) -> list[str]:
    """Demo answers for ``question``, one per pipeline outcome.

    Filters classify nothing, so they only get give-up and correct. A Socratic
    question additionally gets its first misconception (reaches the LLM
    classifier) and a partial answer (resolved as LLM partial).
    """
    presets = [PRESET_GIVE_UP]
    if question.tier == "socratic":
        if question.misconceptions:
            # Misconception is a frozen dataclass, not a mapping.
            presets.append(question.misconceptions[0].wrong_answer)
        partial = _drop_first_key_term(question)
        if partial:
            presets.append(partial)
    presets.append(question.correct_answer)
    return presets


def mock_label_overrides(questions: Iterable[Question]) -> dict[str, str]:
    """Partial preset -> ``"partial"`` for ``MockInferenceClient``.

    The preset text carries no steering word (it is shown as-is on every
    backend), so the mock is told by exact string instead.
    """
    overrides: dict[str, str] = {}
    for question in questions:
        if question.tier == "socratic":
            partial = _drop_first_key_term(question)
            if partial:
                overrides[partial] = "partial"
    return overrides


def _label(text: str) -> str:
    """Bank answers run to full sentences; buttons get a readable stub."""
    if len(text) <= _LABEL_LIMIT:
        return text
    return text[: _LABEL_LIMIT - 1].rstrip() + "…"


def _badge(text: str, bg: str = "#1A1A1D", fg: str = "#E5E5E7") -> str:
    return _BADGE.format(text=escape(str(text)), bg=bg, fg=fg)


# -- callbacks ----------------------------------------------------------------


def _submit() -> None:
    # Streamlit syncs widget values into session_state before callbacks run,
    # so this is the live text area contents, not a stale copy.
    answer = st.session_state.sc_answer_draft
    if not answer.strip():
        return
    session: SocraticSession = st.session_state.sc_session
    st.session_state.sc_last_result = session.submit_answer(answer)
    st.session_state.sc_turn_count += 1
    # Left in place rather than cleared: the graded answer stays on screen next
    # to the verdict it produced. A preset click or an edit replaces it.
    st.session_state.sc_answer_draft = answer


def _fill_draft(text: str) -> None:
    st.session_state.sc_answer_draft = text


def _dismiss_feedback() -> None:
    st.session_state.sc_last_result = None


# -- sections -----------------------------------------------------------------


def render_question_card(question: Question) -> None:
    badges = [_badge(question.id, bg="#7C3AED", fg="#F8FAFC"), _badge(question.cluster)]
    badges.append(_badge(question.tier))
    if question.tier == "socratic" and question.level is not None:
        badges.append(_badge(f"level {question.level}"))
    st.markdown("".join(badges), unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:1.45rem;line-height:1.45;margin:18px 0 6px 0;'>"
        f"{escape(question.question_text)}</div>",
        unsafe_allow_html=True,
    )


def render_answer_input(question: Question) -> None:
    st.text_area("Your answer", key="sc_answer_draft", height=80)
    # Not disabled on an empty draft: the text area only commits its value on
    # blur, so a disabled button would swallow the click that blurs it.
    # ``_submit`` ignores blank answers instead.
    st.button("Submit Answer", type="primary", on_click=_submit)

    st.caption("Demo answers")
    # Stacked rather than columned: bank answers are full sentences and would
    # be truncated past legibility in a narrow column.
    for i, preset in enumerate(build_presets(question)):
        st.button(
            _label(preset),
            key=f"sc_preset_{i}",
            help=preset if len(preset) > _LABEL_LIMIT else None,
            on_click=_fill_draft,
            args=(preset,),
        )


def graded_answer(session: SocraticSession) -> Optional[str]:
    """The answer string the visible verdict was produced from.

    Read back from the log rather than from ``sc_answer_draft``, which a
    preset click or an edit can move on to something else while the feedback
    card is still showing the previous turn.
    """
    event = state.last_answer_event(session.state.history) or {}
    return event.get("answer") or None


def render_feedback(session: SocraticSession, result: TurnResult) -> None:
    history = session.state.history
    answer_event = state.last_answer_event(history) or {}
    verdict = str(answer_event.get("verdict", "wrong"))

    st.divider()
    graded = graded_answer(session)
    if graded:
        st.markdown(
            f"<div style='color:#8A8A90;font-size:0.82rem;margin-bottom:9px;'>"
            f"You answered: {escape(graded)}</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        _badge(verdict, bg=_VERDICT_COLOUR.get(verdict, "#6B7280"), fg="#F8FAFC"),
        unsafe_allow_html=True,
    )
    # result.message is a model-generated hint on most turns; escape it rather
    # than letting the model emit markup into the page.
    st.markdown(
        f"<div style='font-size:1.05rem;line-height:1.5;margin:14px 0;white-space:pre-wrap;'>"
        f"{escape(result.message)}</div>",
        unsafe_allow_html=True,
    )

    question_id = answer_event.get("question_id")
    if question_id:
        cluster = session.bank.get(question_id).cluster
        current, delta = state.mastery_delta(history, cluster)
        if current is not None and delta is not None:
            st.metric(f"Mastery · {cluster}", f"{current:.2f}", f"{delta:+.2f}")

    if result.session_finished:
        st.success("Session complete.")
    else:
        # A hint leaves the same question in play; everything else advanced.
        retry = result.kind == "hint"
        st.button(
            "Try again" if retry else "Next question",
            on_click=_dismiss_feedback,
        )


def render() -> None:
    session: SocraticSession = st.session_state.sc_session
    question: Optional[Question] = session.current_question()
    result: Optional[TurnResult] = st.session_state.sc_last_result

    if question is None:
        if not session.state.question_ids:
            st.info("No questions selected — pick at least one cluster in the sidebar.")
        elif result is not None:
            render_feedback(session, result)
        else:
            st.success("Session complete.")
        return

    render_question_card(question)
    render_answer_input(question)
    if result is not None:
        render_feedback(session, result)
