# Evidence

Two configuration files from the Qualcomm AI Hub Genie bundle that
`socratic_core/inference_client.py` runs on the NPU, and the log of the one
recorded NPU session. The configuration files identify the model build the
NPU path targets and how Genie runs it. The bundle's context binaries are not
in the repo (README.md, Setup).

## npu_bundle_metadata.json

- Model: Qwen3-4B-Instruct-2507, Genie runtime, w4a16, QAIRT 2.45.0 (lines 2-7).
- Target: Snapdragon X Elite, HTP v73 (lines 2144-2151).
- Chat template: `genie.chat_template` (line 2122), read by `load_bundle`
  (`socratic_core/inference_client.py:185-190`).

## npu_bundle_genie_config.json

- Context size: 4096 tokens (line 7).
- Sampler: seed 42, temperature 0.8, top-k 40, top-p 0.95 (lines 14-17).
- Backend: QnnHtp with memory-mapped weights (lines 28, 31); context binaries
  `part1_of_4.bin` to `part4_of_4.bin` (lines 48-53).

## npu_run.json

A session log from an NPU attempt on 2026-09-27, on `s_cell_theory_L1`
(lines 2-4, 208-209). The model never ran:

- Both classifier calls returned a client error in under 1 ms and failed
  closed to logic_error (lines 19-32, 89-102).
- Both hints are the bank fallback; the generator was rejected with
  "client error" (lines 54-57, 123-126).
- The third answer was low_effort from the behavioural layer, and the answer
  was revealed from the bank (lines 163-167, 186-189).

The log does not record the underlying error. The current code logs it as
`client error: <message>` (`socratic_core/classifier_llm.py:334`,
`socratic_core/hint_pipeline.py:270`), added in commit `018b428`
(2026-09-27 17:42 UTC). This log has the bare string; the session finished at
17:22 UTC, before that commit.
