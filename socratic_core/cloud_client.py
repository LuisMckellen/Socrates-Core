"""
cloud_client.py — Qwen3-32B on Groq, behind the same ``generate()`` interface.

Role in the architecture
------------------------
A fourth ``generate()`` backend beside ``InferenceClient`` (NPU),
``LocalCPUClient`` (llama.cpp) and ``MockInferenceClient``. Demo-latency
only: it routes student answers off-device, so it is never the default and
is reached only by an explicit sidebar choice (as the backend, or as the
low-confidence classifier fallback). See CORRECTIONS.md #8.

Prompts arrive already ChatML-formatted by ``inference_client.build_prompt``.
Groq takes chat messages, so ``split_chatml`` turns the string back into
system/user messages; the trailing open assistant block is dropped.

Unlike the local clients, construction *does* raise (``GroqConfigError``)
when no API key is configured: an opted-in cloud backend with no key is a
setup mistake to surface immediately. Runtime failures still never raise;
``generate()`` reports them as ``error``, like every other client.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Optional

API_KEY_NAME = "GROQ_API_KEY"
BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "qwen/qwen3-32b"
# Qwen3 thinks by default; thinking tokens would eat the classifier's cap.
REASONING_EFFORT = "none"
TEMPERATURE = 0.2
TIMEOUT_S = 20.0

_CHATML_BLOCK = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class GroqConfigError(RuntimeError):
    """No Groq API key (or no ``openai`` package) is available."""


def _secret_key() -> Optional[str]:
    """``st.secrets[GROQ_API_KEY]``, or None outside Streamlit / without secrets."""
    try:
        import streamlit as st

        value = st.secrets[API_KEY_NAME]
    except Exception:  # noqa: BLE001 - no streamlit, no secrets.toml, or no such key
        return None
    return str(value) if value else None


def resolve_api_key() -> Optional[str]:
    """st.secrets first, then the environment. None if neither is set."""
    return _secret_key() or os.environ.get(API_KEY_NAME) or None


def groq_available() -> bool:
    return resolve_api_key() is not None


def split_chatml(prompt: str) -> list[dict[str, str]]:
    """ChatML string -> chat messages. Unformatted text becomes one user message."""
    messages = [{"role": role, "content": content} for role, content in _CHATML_BLOCK.findall(prompt)]
    return messages or [{"role": "user", "content": prompt}]


class GroqClient:
    def __init__(self, api_key: Optional[str] = None, model: str = MODEL) -> None:
        key = api_key or resolve_api_key()
        if not key:
            raise GroqConfigError(
                f"Groq backend needs {API_KEY_NAME}: set the environment variable "
                f"or add it to .streamlit/secrets.toml."
            )
        try:
            from openai import OpenAI
        except ImportError as e:
            raise GroqConfigError(f"Groq backend needs the openai package: pip install openai ({e})") from None
        self.model = model
        self.client = OpenAI(api_key=key, base_url=BASE_URL, timeout=TIMEOUT_S, max_retries=1)
        self.last_result: Optional[dict[str, Any]] = None

    def generate(self, prompt: str, max_tokens: int = 256) -> dict:
        """
        Returns:
            {
                "text": str,           # model output
                "ttft_ms": float,      # not streamed: equals total_ms
                "total_ms": float,     # request round trip
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
        if not prompt.strip():
            result["error"] = "empty prompt"
            self.last_result = result
            return result

        try:
            t_start = time.perf_counter()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=split_chatml(prompt),
                max_tokens=max_tokens,
                temperature=TEMPERATURE,
                reasoning_effort=REASONING_EFFORT,
            )
            total_ms = round((time.perf_counter() - t_start) * 1000.0, 1)
            text = _THINK.sub("", response.choices[0].message.content or "").strip()
            usage = getattr(response, "usage", None)
            result["text"] = text
            result["total_ms"] = total_ms
            result["ttft_ms"] = total_ms
            result["tokens_generated"] = int(getattr(usage, "completion_tokens", 0) or len(text.split()))
        except Exception as e:  # noqa: BLE001 - graceful degradation is the contract
            result["text"] = ""
            result["error"] = f"{type(e).__name__}: {e}"

        self.last_result = result
        return result
