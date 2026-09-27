"""Classifier eval: run labelled answers through the real Socratic pipeline on Local CPU.

    python run_eval.py INPUT.jsonl [--out RESULTS.jsonl] [--repo /mnt/e/qualcomm/socrates_core]

Each INPUT line:
    {"answer": "...", "expected_verdict": "wrong", "expected_matched": "ct_hypertrophy",
     "note": "...", "question_id": "s_cell_theory_L1"}      # question_id optional

The session is wired exactly like ui/state.build_session (classifier_fn, verify_fn
and hint_fn all on one LocalCPUClient) but with sessions_dir in a temp directory
and one fresh session per input, started on the target question. Nothing is
written into the repo: session logs go to a temp dir and bytecode is disabled.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True  # keep __pycache__ out of the repo

DEFAULT_REPO = "/mnt/e/qualcomm/socrates_core"
DEFAULT_QUESTION = "s_cell_theory_L1"
BACKEND_LOCAL = "Local CPU (~7s)"  # ui.state.BACKEND_LOCAL, restated to avoid importing streamlit


def percentile(values: list[int], pct: float) -> int:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * pct // 100))  # ceil
    return ordered[int(rank) - 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    from socratic_core.classifier_llm import make_classifier_fn, make_verify_fn
    from socratic_core.hint_pipeline import hint_pipeline
    from socratic_core.local_client import LocalCPUClient
    from socratic_core.question_bank import load_question_bank
    from socratic_core.state_machine import SocraticSession

    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    out_path = Path(args.out) if args.out else Path(args.input).with_suffix(".results.jsonl")

    bank = load_question_bank(Path(args.repo) / "question_bank.json")
    client = LocalCPUClient()
    if client.setup_error:
        print(f"LocalCPUClient failed to load: {client.setup_error}", file=sys.stderr)
        return 2
    sessions_dir = tempfile.mkdtemp(prefix="socratic_eval_sessions_")

    results = []
    for i, row in enumerate(rows, 1):
        session = SocraticSession(
            bank,
            question_ids=[row.get("question_id") or DEFAULT_QUESTION],
            classifier_fn=make_classifier_fn(client),
            verify_fn=make_verify_fn(client),
            hint_fn=lambda q, a, e, *, missing_terms=None, matched_bank_id=None: hint_pipeline(
                q, a, e, client, missing_terms=missing_terms, matched_bank_id=matched_bank_id
            ),
            backend=BACKEND_LOCAL,
            sessions_dir=sessions_dir,
        )
        session.submit_answer(row["answer"])
        ev = next(e for e in reversed(session.state.history) if e.get("event") == "answer")
        result = {
            **row,
            "verdict": ev["verdict"],
            "error_type": ev.get("error_type"),
            "error_source": ev.get("error_source"),
            "matched_bank_id": ev.get("matched_bank_id"),
            "elapsed_ms": ev.get("elapsed_ms"),
        }
        results.append(result)
        print(f"[{i}/{len(rows)}] {result['verdict']:<10} {str(result['matched_bank_id']):<26} "
              f"{result['elapsed_ms']:>6} ms  {row['answer'][:60]}", flush=True)

    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")

    verdict_ok = sum(r["verdict"] == r["expected_verdict"] for r in results)
    # expected_matched "" means the row expects no example match. The session logs
    # no match as "none" (classifier ran) or None (classifier never invoked).
    scored = [r for r in results if r.get("expected_matched")]
    matched_ok = sum(r["matched_bank_id"] == r["expected_matched"] for r in scored)
    abstain_rows = [r for r in results if not r.get("expected_matched")]
    abstain_ok = sum(r["matched_bank_id"] in (None, "none") for r in abstain_rows)
    false_matches = [r for r in abstain_rows if r["matched_bank_id"] not in (None, "none")]
    # Derived view, same rows: a correct answer matched to its natural_correct example
    # is not counted. Kept: matched_bank_id is not "natural_correct", OR expected_verdict
    # is not "correct".
    false_matches_excl = [
        r for r in false_matches
        if r["matched_bank_id"] != "natural_correct" or r["expected_verdict"] != "correct"
    ]
    confusion = Counter((r["verdict"], r["expected_verdict"]) for r in results)
    labels = sorted({v for pair in confusion for v in pair})
    times = [r["elapsed_ms"] for r in results]

    print()
    print(f"total inputs        {len(results)}")
    print(f"verdict accuracy    {verdict_ok}/{len(results)}")
    print(f"matched accuracy    {matched_ok}/{len(scored)}  (rows with non-empty expected_matched)")
    print(f"abstain success     {abstain_ok}/{len(abstain_rows)}  (rows with empty expected_matched; no match logged)")
    print(f"false matches       {len(false_matches)}/{len(abstain_rows)}")
    print(f"false matches (excl. natural_correct on correct rows)  {len(false_matches_excl)}/{len(abstain_rows)}")
    print("confusion (rows = predicted, cols = expected)")
    print("  " + " " * 12 + "".join(f"{lab:>12}" for lab in labels))
    for pred in labels:
        print("  " + f"{pred:<12}" + "".join(f"{confusion.get((pred, exp), 0):>12}" for exp in labels))
    print(f"elapsed_ms p50      {percentile(times, 50)}")
    print(f"elapsed_ms p95      {percentile(times, 95)}")
    print(f"results             {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
