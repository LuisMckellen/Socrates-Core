# Baseline

Frozen 2026-09-27, at the commit that adds this file. It builds on `508b586`:
rule 3c is removed from `socratic_core/hint_pipeline.py` (with its tests in
`tests/test_hint_pipeline.py`), the duplicate `attempt` key is removed from
answer events in `socratic_core/state_machine.py`, and summary lines are added to
`eval/run_eval.py`.

## Tests

`python -m pytest tests/`

| Environment | Passed | Failed | Skipped |
|---|---|---|---|
| Windows venv (`venv/`, Python 3.11.7, no llama-cpp) | 208 | 0 | 1 module (`tests/test_local_client.py`, 2 tests) |
| WSL `socrates` alias (`~/socvenv`, Python 3.11.16, llama_cpp 0.3.35, pytest 9.1.1) | 210 | 0 | 0 |

The gap between the two is the local-client module (README.md:31-34).

Correction to an earlier figure: "210 tests passing" was a `--collect-only`
count in the Windows venv, not a pass count. Before this commit, two tests
failed in both environments:
`tests/test_state_machine.py:775` (`test_key_terms_case_a_turn_result_unchanged`)
and `tests/test_state_machine.py:845` (`test_attempt_number_increments`). The
state machine was still logging a duplicate `attempt` key on answer events.
This commit removes that key, and both tests now pass.

## Eval

Re-scored from the committed results file `eval/socratic_eval_results.jsonl`
(commit `508b586`), using the summary code in `eval/run_eval.py`. The model was
not re-run for this baseline.

Scope: 60 synthetic, hand-labelled answers (`eval/socratic_eval_input.jsonl`),
all on one question (`s_cell_theory_L1`, `eval/run_eval.py:26,65`), classified on
the Local CPU backend (`LocalCPUClient`, `eval/run_eval.py:55`). No NPU run.

| Metric | Result | Rows |
|---|---|---|
| Verdict accuracy | 56/60 | all rows |
| Matched accuracy | 19/21 | rows with a non-empty `expected_matched` |
| Abstain success | 12/39 | rows with an empty `expected_matched`, logged with no match |
| False matches | 27/39 | same 39 rows, logged with a match |
| False matches, excluding correct rows matched to `natural_correct` | 17/39 | same 39 rows |

Elapsed per answer on Local CPU (`elapsed_ms` in the results file): p50 7167 ms,
p95 7388 ms.

## Assets

| Item | Value | Source |
|---|---|---|
| Question bank | 32 questions (16 filter, 16 socratic), 35 misconceptions, 8 clusters | `question_bank.json` |
| CPU model | 2,497,281,120 bytes | `models/qwen3-4b-instruct-2507-q4_k_m.gguf` (not committed, `.gitignore:3`) |
| NPU context binaries | 3,166,048,256 bytes, across `part1_of_4.bin` to `part4_of_4.bin` | Genie bundle, not committed; binaries listed at `evidence/npu_bundle_genie_config.json:48-53` |

## Traces

`evidence/` holds the trace files that verify both changes, with their
comparison results (`evidence/README.md`). Classifier prompt hash (sha256, `s_cell_theory_L1`):
`ff5d42ee50f1beafe1ec9bc9b8184fad2b06380ee16bba872a01265d210a04b8`
(`tests/test_classifier_llm.py:373`).
