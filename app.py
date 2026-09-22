"""app.py — Streamlit front end for Socratic Core.

    streamlit run app.py

All rendering lives inside ``main()`` so that importing this module (as the
smoke tests do) has no side effects. ``streamlit run`` executes the script as
``__main__``, so the guard below still fires.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from ui import debug_panel, sidebar, state, student_view  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="Socratic Core", page_icon="🧠", layout="centered")
    state.init_state()
    sidebar.render()
    student_view.render()
    debug_panel.render(st.session_state.sc_session)


if __name__ == "__main__":
    main()
