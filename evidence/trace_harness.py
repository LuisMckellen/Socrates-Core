"""Re-run of scratch_0a/trace_final.py. Same inputs, same UI wiring; adds
error_type; writes only to the path given as argv[3] (scratch_0a stays untouched).

    python trace_rerun.py <repo_path> <reps> <out_json>
"""
import json, sys, tempfile
from pathlib import Path
REPO = Path(sys.argv[1]); reps = int(sys.argv[2]); OUT = Path(sys.argv[3])
sys.path.insert(0, str(REPO))
from socratic_core.classifier_llm import make_classifier_fn, make_verify_fn
from socratic_core.hint_pipeline import MAX_TOKENS as HINT_MAX_TOKENS, hint_pipeline
from socratic_core.local_client import LocalCPUClient
from socratic_core.question_bank import load_question_bank
from socratic_core.state_machine import SocraticSession
QID = "s_cell_theory_L1"
bank = load_question_bank(REPO / "question_bank.json"); q = bank.get(QID)
TRACES = [("Trace 1", q.misconceptions[0].wrong_answer), ("Trace 2", q.misconceptions[1].wrong_answer),
          ("Trace 3", "Cells come from pre-existing things i guess"),
          ("Para A", "the damaged tissue just gets bigger to fill the gap"),
          ("Para B", "cells near the wound soak up nutrients until they cover the hole"),
          ("Para C", "new cells appear at the wound site from the fluid and matrix there")]
class Recording:
    def __init__(self, inner): self.inner, self.calls = inner, []
    def generate(self, prompt, max_tokens=256):
        r = self.inner.generate(prompt, max_tokens=max_tokens)
        self.calls.append({"max_tokens": max_tokens, "prompt": prompt, "text": r.get("text")}); return r
base = LocalCPUClient(); assert base.setup_error is None, base.setup_error
out = []
with tempfile.TemporaryDirectory() as d:
    for name, answer in TRACES:
        for rep in range(reps):
            client = Recording(base); results = []
            def hint_fn(qq, a, e, *, missing_terms=None, matched_bank_id=None):
                r = hint_pipeline(qq, a, e, client, missing_terms=missing_terms, matched_bank_id=matched_bank_id)
                results.append(r); return r
            s = SocraticSession(bank, question_ids=[QID], classifier_fn=make_classifier_fn(client),
                                verify_fn=make_verify_fn(client), hint_fn=hint_fn, sessions_dir=d)
            s.submit_answer(answer)
            ans = [h for h in s.state.history if h["event"] == "answer"][-1]
            rec = {"input": name, "rep": rep, "verdict": ans["verdict"], "error_type": ans.get("error_type"),
                   "error_source": ans["error_source"], "matched_bank_id": ans["matched_bank_id"],
                   "source": results[-1]["source"] if results else None, "rejections": ans["rejections"],
                   "replies": [c["text"] for c in client.calls if c["max_tokens"] == HINT_MAX_TOKENS],
                   "final_hint": ans["hint_text"], "answer_keys": list(ans)}
            out.append(rec); print(json.dumps(rec, ensure_ascii=False), flush=True)
OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
