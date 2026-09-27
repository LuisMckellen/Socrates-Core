# Evidence

Two configuration files from the Qualcomm AI Hub Genie bundle that
`socratic_core/inference_client.py` runs on the NPU. They identify the model
build the NPU path targets and how Genie runs it. The bundle's context
binaries are not in the repo (README.md, Setup).

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
