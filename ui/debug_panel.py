"""debug_panel.py — what the pipeline did on the last turn.

Reads ``session.state.history`` only. Every field is optional, so every read
goes through ``.get()``: a filter turn logs no error_type, only a model call
logs a per-call latency, and only an escalating filter logs a target question.
"""

from __future__ import annotations

import streamlit as st

from socratic_core.state_machine import SocraticSession

from . import state


def _render_last_turn(history: list[dict]) -> None:
    answer = state.last_answer_event(history)
    if answer is None:
        st.caption("No answers submitted yet.")
        return

    turn = state.last_turn_slice(history)
    classify = state.find_event(turn, "llm_classify") or {}
    hint = state.find_event(turn, "llm_hint") or {}
    error_source = answer.get("error_source")

    left, middle, right = st.columns(3)
    left.metric("error_type", str(answer.get("error_type") or "—"))
    middle.metric("error_source", str(error_source or "—"))
    right.metric("llm_called", "yes" if state.llm_called(error_source) else "no")

    st.markdown(f"**Resolved at** · {state.resolved_at(error_source)}")

    st.markdown(f"**elapsed_ms** · {answer.get('elapsed_ms')} ms")
    classify_ms = classify.get("elapsed_ms")
    hint_ms = hint.get("elapsed_ms")
    parts = []
    if classify_ms is not None:
        parts.append(f"classify {classify_ms} ms")
    if hint_ms is not None:
        parts.append(f"hint {hint_ms} ms")
    if parts:
        st.markdown(f"**model calls** · {' · '.join(parts)}")

    matched = answer.get("key_terms_matched")
    missing = answer.get("key_terms_missing")
    if matched is not None or missing is not None:
        st.markdown(
            f"**key_terms** · matched `{matched or []}` · missing `{missing or []}`"
        )

    tokens = answer.get("disengagement_tokens") or []
    st.markdown(
        f"**disengagement** · flag `{bool(answer.get('disengagement_flag'))}` · "
        f"tokens `{tokens}`"
    )

    escalation = state.find_event(turn, "escalation_choice")
    if escalation and escalation.get("from_filter_id") and escalation.get("to_question_id"):
        st.markdown(
            f"**escalation_choice** · `{escalation['from_filter_id']}` → "
            f"`{escalation['to_question_id']}` ({escalation.get('reason', '—')})"
        )

    gap = state.find_event(turn, "topic_gap")
    if gap and gap.get("cluster") and gap.get("reason"):
        st.markdown(f"**topic_gap** · `{gap['cluster']}` ({gap['reason']})")

    snapshots = state.mastery_snapshots(history)
    if snapshots:
        st.markdown("**mastery**")
        st.json(snapshots[-1], expanded=False)


def render(session: SocraticSession) -> None:
    with st.expander("🔍 Pipeline internals", expanded=False):
        history = session.state.history
        _render_last_turn(history)

        rows = state.answer_rows(history)
        if rows:
            st.markdown("**All turns**")
            st.dataframe(rows, hide_index=True)
