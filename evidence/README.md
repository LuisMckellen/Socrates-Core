# Evidence

The NPU Genie bundle (context binaries, ~3.16 GB total, BASELINE.md:55) is
not committed. It is produced by Qualcomm AI Hub for the Snapdragon X Elite
target and downloaded to a local folder next to the repo
(`socratic_core/inference_client.py:73-76`, override with
`SOCRATIC_MODEL_DIR`). The two JSON files in this folder are copied byte for
byte from that bundle, so the build can be cited without shipping the
binaries. No NPU run of the pipeline has been recorded (BASELINE.md).

## npu_bundle_metadata.json

The bundle's AI Hub metadata.

- Model: Qwen3-4B-Instruct-2507 (lines 2-3), Genie runtime (line 4),
  w4a16 precision (line 5).
- Toolchain: QAIRT 2.45.0 (line 7).
- Target: Snapdragon X Elite, HTP v73, reference device Snapdragon X Elite
  CRD (lines 2144-2151).
- Chat template (from line 2122). `inference_client.load_bundle` reads it
  from here at runtime; `DEFAULT_CHAT_TEMPLATE`
  (`socratic_core/inference_client.py:94-102`) is the fallback copy.

## npu_bundle_genie_config.json

The Genie runtime config that `genie-t2t-run -c` loads.

- Context size 4096 (line 7).
- Sampler: seed 42, temperature 0.8 (lines 14-15). The CPU and Groq
  backends run at 0.2, so NPU output is not directly comparable to them.
- Backend QnnHtp (line 28) with `use-mmap` enabled (line 31).
- The four context binaries, `part1_of_4.bin` to `part4_of_4.bin`
  (lines 48-53). `inference_client.load_bundle` reads this list rather than
  assuming it.
