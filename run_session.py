"""
run_session.py — interactive terminal loop for the Socratic tutor.

Wires the real classifier and hint pipeline into ``SocraticSession`` and
drives it with ``input()``. This is the only place the LLM-facing adapters
live; the core modules stay injectable and model-free.

    python run_session.py                          # first question, mock client
    python run_session.py --question-id bio_003    # one question
    python run_session.py --question-id bio_003 --client local   # real model on CPU
    python run_session.py --question-id bio_001 bio_002 --client npu

The session log is written by the state machine to ``sessions/`` (or
``SOCRATIC_SESSIONS_DIR``); nothing here writes it directly.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from socratic_core.classifier_llm import make_classifier_fn  # noqa: E402
from socratic_core.hint_pipeline import hint_pipeline  # noqa: E402
from socratic_core.inference_client import InferenceClient  # noqa: E402
from socratic_core.local_client import LocalCPUClient  # noqa: E402
from socratic_core.mock_client import MockInferenceClient  # noqa: E402
from socratic_core.question_bank import load_question_bank  # noqa: E402
from socratic_core.state_machine import SocraticSession  # noqa: E402


def main(argv: Optional[list[str]] = None, input_fn: Callable[[str], str] = input) -> dict:
    parser = argparse.ArgumentParser(description="Run one Socratic tutoring session.")
    parser.add_argument(
        "--question-id",
        nargs="+",
        metavar="ID",
        help="question id(s) to ask, in order (default: first question in the bank)",
    )
    parser.add_argument("--client", choices=("mock", "local", "npu"), default="mock")
    args = parser.parse_args(argv)

    bank = load_question_bank()
    if args.client == "mock":
        client = MockInferenceClient()
    elif args.client == "local":
        client = LocalCPUClient()
    else:
        client = InferenceClient()
    try:
        session = SocraticSession(
            bank,
            question_ids=args.question_id or bank.ids()[:1],
            classifier_fn=make_classifier_fn(client),
            hint_fn=lambda q, a, e: hint_pipeline(q, a, e, client)["hint"],
        )
    except KeyError as e:
        parser.error(f"{e.args[0]} (known ids: {', '.join(bank.ids())})")

    t0 = time.perf_counter()
    turns = 0
    while not session.state.finished:
        q = session.current_question()
        print(f"\n[{q.id}] {q.question_text}")
        try:
            answer = input_fn("> ")
        except (EOFError, StopIteration):
            print("\n(input ended; session left unfinished)")
            break
        turns += 1
        r = session.submit_answer(answer)
        if r.kind == "correct":
            print("Correct.")
        elif r.kind == "hint":
            print(f"Hint ({r.error_type}, attempt {r.attempt}): {r.message}")
        else:
            print(f"Let's move on.\n{r.message}")

    summary = {
        "solved_question_ids": list(session.state.solved_question_ids),
        "stuck_question_ids": list(session.state.stuck_question_ids),
        "turns": turns,
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }
    print("\n--- session summary ---")
    print(f"solved:  {summary['solved_question_ids']}")
    print(f"stuck:   {summary['stuck_question_ids']}")
    print(f"turns:   {summary['turns']}")
    print(f"elapsed: {summary['elapsed_s']}s")
    print(f"log:     {session.log_path}")
    return summary


if __name__ == "__main__":
    main()
