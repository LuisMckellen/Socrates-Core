# Evidence

Trace runs from 2026-09-27 that verify two changes: removing rule 3c from the
hint validator, and removing the duplicate `attempt` key from answer events.
Every run uses question `s_cell_theory_L1` on the Local CPU backend (GGUF
Q4_K_M via llama-cpp-python, the fallback runtime, not the NPU) with the same
wiring as `ui/state.py` `build_session`. There are six inputs (Trace 1–3,
Para A–C) and 3 reps each, so 18 runs per file. The exception is
`trace_rule3c_original_trace2.json`, which holds Trace 2 only (3 runs). The
classifier prompt hash
(sha256, `s_cell_theory_L1`) was
`ff5d42ee50f1beafe1ec9bc9b8184fad2b06380ee16bba872a01265d210a04b8` for both
runs, the value asserted at `tests/test_classifier_llm.py:373`.

## trace_harness.py

The script that produced the files below. It uses the same six inputs and the
same wiring as `scratch_0a/trace_final.py`, and records `error_type` as well.
It writes only to the output path it is given. Run it from the repo root in
the WSL `socrates` environment (`~/socvenv`, llama_cpp 0.3.35):
`python evidence/trace_harness.py . 3 <out.json>`. This copy adds two things
to the script that produced `trace_rule3c_postfix.json`: the output-path
argument and the `answer_keys` field. That earlier script was lost when WSL
restarted and cleared `/tmp`. Its logic was otherwise the same.

## trace_rule3c_original_trace2.json

3 records: Trace 2 before rule 3c was removed. This is a byte-for-byte copy of
`scratch_0a/trace_final_Trace_2.json`, produced by `scratch_0a/trace_final.py`.
On all 3 reps, rule 3c rejected both generated hints ("names a key term on the
targeted path"), so the hint source was `template`, the bank fallback. That is
6 of 6 hints rejected.

## trace_rule3c_postfix.json

18 records, run after rule 3c was removed. Produced by the earlier version of
the harness (`python /tmp/rule3c/trace_rerun.py . 3` under `socrates`); this is
a byte-for-byte copy of its output. Compared with the pre-removal run
(`scratch_0a/trace_final.json`, 18 records), Trace 2's hint source changed from
`template` to `llm` on all 3 reps. Verdict, error_source and matched_bank_id
stayed the same on all 18 runs, and hint source stayed the same on every
other input. The pre-removal file does not record `error_type`, so that field
has no earlier value to compare against.

## trace_attemptfix.json

18 records, run after the duplicate `attempt` key was removed. Produced by
`python evidence/trace_harness.py . 3 <out.json>` under `socrates`. Compared
with `trace_rule3c_postfix.json`, verdict, error_type, error_source,
matched_bank_id and hint source are identical on all 18 runs, and so is the
final hint text. Each record also carries `answer_keys`: on every run the
answer event has `attempt_number` and no `attempt`.

## npu_bundle_metadata.json, npu_bundle_genie_config.json

The NPU Genie bundle itself (context binaries, ~3.16 GB total,
BASELINE.md:55) is not committed. It is produced by Qualcomm AI Hub for the
Snapdragon X Elite target (w4a16, QnnHtp) and downloaded to a local folder
next to the repo (`socratic_core/inference_client.py:73-76`). The two JSON
files in this folder are the bundle's metadata and Genie runtime config,
copied byte for byte for citation.
