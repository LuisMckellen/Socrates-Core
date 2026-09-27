# Results

Measured results for the code in this repository. README.md quotes these
numbers.

## Tests

`python -m pytest tests/`

| Environment | Passed | Failed | Skipped |
|---|---|---|---|
| Windows venv (Python 3.11.7, no llama-cpp-python) | 208 | 0 | 1 module (`tests/test_local_client.py`, 2 tests) |
| WSL venv (Python 3.11.16, llama-cpp-python 0.3.35, pytest 9.1.1) | 210 | 0 | 0 |

The difference between the two is `tests/test_local_client.py`, which skips
itself at import time when llama_cpp is not installed
(`tests/test_local_client.py:22-25`).

## Eval

Scores come from `eval/socratic_eval_results.jsonl`, computed with the
summary code in `eval/run_eval.py:90-121`.

Scope: 60 synthetic, hand-labelled answers (`eval/socratic_eval_input.jsonl`),
all on one question (`s_cell_theory_L1`, `eval/run_eval.py:26,65`), classified
on the Local CPU backend (`LocalCPUClient`, `eval/run_eval.py:55`). The eval
does not run on the NPU.

| Metric | Result | Rows |
|---|---|---|
| Verdict accuracy | 56/60 | all rows |
| Matched accuracy | 19/21 | rows with a non-empty `expected_matched` |
| Abstain success | 12/39 | rows with an empty `expected_matched`, logged with no match |
| False matches | 27/39 | same 39 rows, logged with a match |
| False matches, excluding correct rows matched to `natural_correct` | 17/39 | same 39 rows |

Elapsed per answer on Local CPU (`elapsed_ms` in the results file): p50
7167 ms, p95 7388 ms.

## Assets

| Item | Value | Location |
|---|---|---|
| Question bank | 32 questions (16 filter, 16 socratic), 35 misconceptions, 8 clusters | `question_bank.json` |
| CPU model | 2,497,281,120 bytes | `models/qwen3-4b-instruct-2507-q4_k_m.gguf` (not in the repo, `.gitignore:3`) |
| NPU context binaries | 3,166,048,256 bytes, across `part1_of_4.bin` to `part4_of_4.bin` | Genie bundle (not in the repo); names listed at `evidence/npu_bundle_genie_config.json:48-53` |

## Classifier prompt hash

The classifier prompt for `s_cell_theory_L1` is pinned by
`tests/test_classifier_llm.py:373` to
`ff5d42ee50f1beafe1ec9bc9b8184fad2b06380ee16bba872a01265d210a04b8`.
