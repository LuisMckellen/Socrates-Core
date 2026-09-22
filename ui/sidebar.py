"""sidebar.py — backend switch, cluster filter and session reset.

Every control here re-instantiates the session, so each one is guarded: the
backend radio needs an explicit confirmation tick (switching to the local
model loads several GB), and the cluster multiselect only rebuilds when the
selection actually changed.
"""

from __future__ import annotations

import streamlit as st

from . import state


def _sync_backend() -> None:
    """Apply a pending backend switch once the user confirms it."""
    choice = st.session_state.sc_backend_choice
    if choice == st.session_state.sc_backend:
        # Nothing pending: drop a stale tick so the next switch asks again.
        st.session_state.pop("sc_backend_confirm", None)
        return

    st.warning(f"Switch to **{choice}**? This restarts the session.", icon="⚠️")
    if st.checkbox("Confirm backend switch", key="sc_backend_confirm"):
        st.session_state.sc_backend = choice
        state.rebuild_session()
        st.rerun()


def _sync_clusters() -> None:
    selected = list(st.session_state.sc_cluster_choice)
    if set(selected) != set(st.session_state.sc_clusters):
        st.session_state.sc_clusters = selected
        state.rebuild_session()
        st.rerun()


def render() -> None:
    bank = state.get_bank()
    with st.sidebar:
        st.title("Socratic Core")
        st.caption("Qwen3-4B · Snapdragon X Elite NPU")
        st.divider()

        st.session_state.setdefault("sc_backend_choice", st.session_state.sc_backend)
        st.radio("Backend", state.BACKEND_LABELS, key="sc_backend_choice")
        _sync_backend()

        st.divider()

        st.session_state.setdefault("sc_cluster_choice", st.session_state.sc_clusters)
        st.multiselect("Clusters", state.all_clusters(bank), key="sc_cluster_choice")
        _sync_clusters()

        st.divider()

        if st.button("Reset session"):
            state.rebuild_session()
            st.rerun()

        session_id = st.session_state.sc_session.state.session_id
        turns = st.session_state.sc_turn_count
        st.caption(f"session …{session_id[-6:]} · {turns} turn{'' if turns == 1 else 's'}")
