"""student_view.py — the question, the answer box and the feedback card.

Everything that mutates the session runs in an ``on_click`` callback. Streamlit
runs callbacks *before* the script re-executes, which is the only point at
which ``sc_answer_draft`` may be written: once the text area has been
instantiated on a given pass, assigning to its key raises.

The page has a mode, derived from the last TurnResult (``render_mode``). A
generated hint is the prompt for the *next* answer, so on a "probe" it renders
above the answer box as part of the question card; the feedback card below
only carries the verdict on the answer just given.
"""

from __future__ import annotations

from html import escape
from typing import Optional

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

CELL_THEORY_L1 = "s_cell_theory_L1"
# Contradicts itself; a manual string, not a bank misconception.
CELL_THEORY_L1_CONTRADICTION = "Cells divide because they don't divide"
# Both key terms, not a bank variant: Case A, so the demo exercises verify_fn.
CELL_THEORY_L1_CORRECT = "Pre-existing cells undergo division to form new tissue"

MODE_BANK = "bank"
MODE_PROBE = "probe"
MODE_ESCALATE = "escalate"
MODE_FINISHED = "finished"
# TurnResult kinds that moved the session on to another question.
ADVANCE_KINDS = frozenset({"correct", "escalated", "filter_escalated", "filter_missed"})


def build_presets(question: Question) -> list[str]:
    """Demo answers for ``question``: the fill-in buttons under the answer box.

    Filters classify nothing, so they get give-up and correct (2).
    ``s_cell_theory_L1``, the demo's escalation target, gets give-up, its first
    bank misconception verbatim, a self-contradicting answer (a manual string,
    not a bank misconception) and a Case A correct answer (4). Any other
    Socratic question gets give-up, its first misconception and the bank's
    correct answer (3). The list is the same on every attempt: "Revise" on a
    probe is the answer box's label, not a preset.

    The partial path is exercised by test_classifier_llm and by the state
    machine tests. There is no partial preset because _drop_first_key_term
    produced ungrammatical text.
    """
    # Misconception is a frozen dataclass, not a mapping.
    if question.id == CELL_THEORY_L1:
        return [
            PRESET_GIVE_UP,
            question.misconceptions[0].wrong_answer,
            CELL_THEORY_L1_CONTRADICTION,
            CELL_THEORY_L1_CORRECT,
        ]
    presets = [PRESET_GIVE_UP]
    if question.tier == "socratic" and question.misconceptions:
        presets.append(question.misconceptions[0].wrong_answer)
    presets.append(question.correct_answer)
    return presets


def render_mode(result: Optional[TurnResult]) -> str:
    """"bank" | "probe" | "escalate" | "finished", from the last TurnResult.

    ``session_finished`` is checked first: the final turn also carries an
    advancing kind.
    """
    if result is None:
        return MODE_BANK
    if result.session_finished:
        return MODE_FINISHED
    if result.kind == "hint":
        return MODE_PROBE
    return MODE_ESCALATE  # every ADVANCE_KINDS member


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
    result: Optional[TurnResult] = st.session_state.sc_last_result
    # The kept draft answered the previous question; a new one starts blank.
    if result is not None and result.kind in ADVANCE_KINDS:
        st.session_state.sc_answer_draft = ""
    st.session_state.sc_last_result = None


# -- sections -----------------------------------------------------------------


def render_question_card(question: Question, probe: Optional[dict] = None) -> None:
    """The bank question; with ``probe`` (the last answer event), the hint too.

    On a probe the student is revising, not starting fresh: the bank question
    is dimmed, their answer is quoted back, and the hint is set apart from
    question_text by a left border, a tinted background and italics.
    """
    badges = [_badge(question.id, bg="#7C3AED", fg="#F8FAFC"), _badge(question.cluster)]
    badges.append(_badge(question.tier))
    if question.tier == "socratic" and question.level is not None:
        badges.append(_badge(f"level {question.level}"))
    st.markdown("".join(badges), unsafe_allow_html=True)
    if probe is None:
        st.markdown(
            f"<div style='font-size:1.45rem;line-height:1.45;margin:18px 0 6px 0;'>"
            f"{escape(question.question_text)}</div>",
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        f"<div style='font-size:1.05rem;line-height:1.45;color:#8A8A90;margin:14px 0 12px 0;'>"
        f"{escape(question.question_text)}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='color:#8A8A90;font-size:0.9rem;margin-bottom:10px;'>"
        f"You said: “{escape(str(probe.get('answer', '')))}”</div>",
        unsafe_allow_html=True,
    )
    # hint_text is model-generated; escaped like any other model output.
    st.markdown(
        f"<div style='border-left:4px solid #7C3AED;background:rgba(124,58,237,0.08);"
        f"padding:12px 16px;border-radius:0 8px 8px 0;font-style:italic;"
        f"font-size:1.25rem;line-height:1.45;margin:0 0 10px 0;'>"
        f"{escape(str(probe.get('hint_text', '')))}</div>",
        unsafe_allow_html=True,
    )


def submit_label(mode: str) -> str:
    return "Revise" if mode == MODE_PROBE else "Submit Answer"


def render_answer_input(question: Question, mode: str = MODE_BANK) -> None:
    st.text_area("Your answer", key="sc_answer_draft", height=80)
    # Not disabled on an empty draft: the text area only commits its value on
    # blur, so a disabled button would swallow the click that blurs it.
    # ``_submit`` ignores blank answers instead.
    st.button(submit_label(mode), type="primary", on_click=_submit)

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


def render_feedback(session: SocraticSession, result: TurnResult, mode: str) -> None:
    """The verdict on the answer just given, plus mastery.

    On a probe the hint and the quoted answer are in the question card
    instead, and there is no button: dismissing would hide the probe.
    Otherwise ``result.message`` is "Correct.", a filter's note, or the
    two-line reveal after max attempts.
    """
    history = session.state.history
    answer_event = state.last_answer_event(history) or {}
    verdict = str(answer_event.get("verdict", "wrong"))

    st.divider()
    graded = graded_answer(session)
    if graded and mode != MODE_PROBE:
        st.markdown(
            f"<div style='color:#8A8A90;font-size:0.82rem;margin-bottom:9px;'>"
            f"You answered: {escape(graded)}</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        _badge(verdict, bg=_VERDICT_COLOUR.get(verdict, "#6B7280"), fg="#F8FAFC"),
        unsafe_allow_html=True,
    )
    if mode != MODE_PROBE:
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

    if mode == MODE_FINISHED:
        st.success("Session complete.")
    elif mode == MODE_ESCALATE:
        st.button("Next question", on_click=_dismiss_feedback)


def render() -> None:
    session: SocraticSession = st.session_state.sc_session
    question: Optional[Question] = session.current_question()
    result: Optional[TurnResult] = st.session_state.sc_last_result
    mode = render_mode(result)

    if question is None:
        if not session.state.question_ids:
            st.info("No questions selected — pick at least one cluster in the sidebar.")
        elif result is not None:
            render_feedback(session, result, mode)
        else:
            st.success("Session complete.")
        return

    # Escalate: the verdict belongs to the previous question, so it comes first.
    if mode == MODE_ESCALATE:
        render_feedback(session, result, mode)
    probe: Optional[dict] = None
    if mode == MODE_PROBE:
        probe = dict(state.last_answer_event(session.state.history) or {})
        # The hint path always fills hint_text; result.message is the same string.
        probe["hint_text"] = probe.get("hint_text") or result.message
    render_question_card(question, probe)
    render_answer_input(question, mode)
    if mode == MODE_PROBE:
        render_feedback(session, result, mode)
