"""
local_client.py — run the same Qwen3 model on the CPU via llama.cpp.

Role in the architecture
------------------------
A third ``generate()`` backend beside ``InferenceClient`` (NPU, Genie) and
``MockInferenceClient``. Same weights as the compiled NPU binary, different
runtime and quantisation (GGUF Q4_K_M), so the real model can be exercised
on an x86-64 development machine where the Genie runtime cannot run.

Prompts arrive already chat-formatted: every caller goes through
``inference_client.build_prompt()``, which applies the ChatML template the
NPU bundle uses. This client therefore runs a raw completion and stops at
``<|im_end|>``; applying a template here would wrap the prompt twice.

Construction never raises. A missing file or a failed load is remembered and
reported as ``error`` on every ``generate()`` call, matching the other clients.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

DEFAULT_MODEL_PATH = "/home/rickt/models/qwen3-4b-instruct-2507-q4_k_m.gguf"
STOP_SEQUENCES = ("<|im_end|>",)
TEMPERATURE = 0.2
# Physical cores on the Ryzen 7 7700X dev box. Measured against 12: 8 was
# marginally faster on every call (within noise); decode is bandwidth-bound.
N_THREADS = 8


class LocalCPUClient:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, n_ctx: int = 4096) -> None:
        path = Path(model_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / path
        self.model_path = path
        self.n_ctx = n_ctx
        self.llm: Any = None
        self.setup_error: Optional[str] = None
        self.last_result: Optional[dict[str, Any]] = None

        if not path.is_file():
            self.setup_error = f"model file not found: {path}"
            return
        try:
            from llama_cpp import Llama

            self.llm = Llama(model_path=str(path), n_ctx=n_ctx, n_gpu_layers=0, n_threads=N_THREADS, verbose=False)
        except Exception as e:  # noqa: BLE001 - surfaced via generate(), never raised
            self.setup_error = f"{type(e).__name__}: {e}"

    def generate(self, prompt: str, max_tokens: int = 256) -> dict:
        """
        Returns:
            {
                "text": str,           # model output
                "ttft_ms": float,      # time to first token
                "total_ms": float,     # total generation time
                "tokens_generated": int,
                "error": str | None    # None on success
            }
        Never raises for runtime failures; ``error`` carries the reason.
        """
        result: dict[str, Any] = {
            "text": "",
            "ttft_ms": 0.0,
            "total_ms": 0.0,
            "tokens_generated": 0,
            "error": None,
        }
        if self.setup_error or self.llm is None:
            result["error"] = self.setup_error or "client not initialised"
            self.last_result = result
            return result
        if not prompt.strip():
            result["error"] = "empty prompt"
            self.last_result = result
            return result

        try:
            t_start = time.perf_counter()
            first_token_at: Optional[float] = None
            pieces: list[str] = []
            for chunk in self.llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=TEMPERATURE,
                stop=list(STOP_SEQUENCES),
                stream=True,
            ):
                piece = chunk["choices"][0]["text"]
                if first_token_at is None and piece:
                    first_token_at = time.perf_counter()
                pieces.append(piece)
            t_end = time.perf_counter()

            text = "".join(pieces).strip()
            result["text"] = text
            result["total_ms"] = round((t_end - t_start) * 1000.0, 1)
            result["ttft_ms"] = round(((first_token_at or t_end) - t_start) * 1000.0, 1)
            result["tokens_generated"] = len(self.llm.tokenize(text.encode("utf-8"), add_bos=False))
        except Exception as e:  # noqa: BLE001 - graceful degradation is the contract
            result["text"] = ""
            result["error"] = f"{type(e).__name__}: {e}"

        self.last_result = result
        return result
