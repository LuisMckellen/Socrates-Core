"""
inference_client.py — wrapper around the Genie (QAIRT) runtime for Qwen3 on the NPU.

Role in the architecture
------------------------
The only module that talks to the model. ``classifier_llm`` and
``hint_pipeline`` call ``InferenceClient.generate()`` and nothing else. It
never decides anything about the dialogue; it just turns a prompt into text
and reports how long that took.

How it runs the model
---------------------
The AI Hub export for this model is a *Genie bundle*: ``genie_config.json``
plus four context binaries (``part1_of_4.bin`` .. ``part4_of_4.bin``), a
tokenizer and an HTP backend config. The documented way to execute such a
bundle is the Genie SDK's ``genie-t2t-run`` executable, which is what the
``qai_hub_models`` package itself shells out to when it benchmarks Genie
bundles (``qai_hub_models/utils/llm/genie/device_scripts/run_android.py``).
This client does the same via ``subprocess``:

    genie-t2t-run -c genie_config.json --prompt_file <tmp> --profile <tmp.json>

run with the bundle folder as the working directory (every path inside
``genie_config.json`` is relative to it).

Everything Genie-specific is isolated in the ``_run_genie``/``_parse_*``
helpers so the transport can be swapped (e.g. for a persistent in-process
runtime) without touching callers.

Known limitations (flagged, not hidden)
---------------------------------------
* Each ``generate()`` spawns a fresh ``genie-t2t-run`` process, which
  re-loads ~3 GB of context binaries. Wall-clock latency therefore includes
  model load. The Genie profile JSON reports TTFT *excluding* load, and that
  value is preferred when available.
* ``genie-t2t-run`` only exists for Snapdragon targets (the HTP backend needs
  the NPU). On an x86-64 development host this client will report a clean
  ``error`` rather than crash.
* ``max_tokens`` is recorded but not yet enforced — see the TODO in
  ``_apply_max_tokens``.

Configuration (no hardcoded paths in code)
------------------------------------------
``SOCRATIC_MODEL_DIR``   bundle folder; default is the AI Hub export folder
                         next to this project (see ``DEFAULT_MODEL_DIR``).
``SOCRATIC_GENIE_BIN``   full path to ``genie-t2t-run[.exe]``. If unset, the
                         client looks under ``QNN_SDK_ROOT``/``QAIRT_SDK_ROOT``
                         (``bin/*/genie-t2t-run*``) and then on ``PATH``.
``SOCRATIC_GENIE_TIMEOUT_S``  per-call timeout in seconds (default 180).
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

ENV_MODEL_DIR = "SOCRATIC_MODEL_DIR"
ENV_GENIE_BIN = "SOCRATIC_GENIE_BIN"
ENV_TIMEOUT_S = "SOCRATIC_GENIE_TIMEOUT_S"
ENV_SDK_ROOTS = ("QNN_SDK_ROOT", "QAIRT_SDK_ROOT")

# Default = the AI Hub export folder as it sits next to this project.
# Override with SOCRATIC_MODEL_DIR; nothing else in the code assumes this path.
DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "qwen3_4b_instruct_2507-genie-w4a16-qualcomm_snapdragon_x_elite"
)
DEFAULT_TIMEOUT_S = 180.0

GENIE_CONFIG_NAME = "genie_config.json"
METADATA_NAME = "metadata.json"
GENIE_BIN_NAME = "genie-t2t-run"

# genie-t2t-run prints the completion between these markers (source:
# qai_hub_models/utils/llm/genie/jobs.py::_extract_model_output).
BEGIN_MARKER = "[BEGIN]:"
END_MARKER = "[END]"


# -- prompt formatting ---------------------------------------------------------

# Fallback chat template. The bundle's metadata.json carries the same values
# under "genie" -> "chat_template"; ``InferenceClient.bundle.chat_template``
# is the authoritative copy once a bundle is loaded.
DEFAULT_CHAT_TEMPLATE: dict[str, str] = {
    "system_prefix": "<|im_start|>system\n",
    "system_suffix": "<|im_end|>\n",
    "user_prefix": "<|im_start|>user\n",
    "user_suffix": "<|im_end|>\n",
    "assistant_prefix": "<|im_start|>assistant\n",
    "assistant_suffix": "<|im_end|>\n",
    "default_system_prompt": "You are a helpful AI assistant.",
}


def build_prompt(
    user: str,
    system: Optional[str] = None,
    template: Optional[dict[str, str]] = None,
) -> str:
    """Wrap a user message in the model's chat template.

    Produces the same layout as the bundle's ``sample_prompt.txt``:
    system block, user block, then an open assistant block for the model to
    fill. Callers pass the *result* of this to ``generate()``.
    """
    t = {**DEFAULT_CHAT_TEMPLATE, **(template or {})}
    system_text = system if system is not None else t["default_system_prompt"]
    return (
        f"{t['system_prefix']}{system_text}{t['system_suffix']}"
        f"{t['user_prefix']}{user}{t['user_suffix']}"
        f"{t['assistant_prefix']}"
    )


# -- bundle description --------------------------------------------------------


@dataclass
class GenieBundle:
    """What we know about the compiled model folder after validation."""

    model_dir: Path
    config_path: Path
    ctx_bins: list[Path]
    tokenizer_path: Path
    context_size: int
    chat_template: dict[str, str] = field(default_factory=dict)
    qairt_version: Optional[str] = None


class BundleError(RuntimeError):
    """The model folder is missing or does not look like a Genie bundle."""


def load_bundle(model_dir: str | os.PathLike[str] | None = None) -> GenieBundle:
    """Validate the bundle folder and resolve the 4-part binary list.

    The list of context binaries is read from ``genie_config.json`` rather
    than assumed, so a re-export with a different split still loads.
    """
    root = Path(model_dir or os.environ.get(ENV_MODEL_DIR) or DEFAULT_MODEL_DIR)
    if not root.is_dir():
        raise BundleError(f"model dir not found: {root} (set {ENV_MODEL_DIR})")

    config_path = root / GENIE_CONFIG_NAME
    if not config_path.is_file():
        raise BundleError(f"{GENIE_CONFIG_NAME} missing in {root}")
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        dialog = cfg["dialog"]
        engine = dialog["engine"]
        bin_names: list[str] = engine["model"]["binary"]["ctx-bins"]
        tokenizer_rel: str = dialog["tokenizer"]["path"]
        context_size = int(dialog["context"]["size"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
        raise BundleError(f"{config_path}: unexpected layout: {e!r}") from None

    ctx_bins = [root / name for name in bin_names]
    missing = [p.name for p in ctx_bins if not p.is_file()]
    if missing:
        raise BundleError(f"context binaries missing in {root}: {missing}")
    if not ctx_bins:
        raise BundleError(f"{config_path}: 'ctx-bins' is empty")

    tokenizer_path = root / tokenizer_rel
    if not tokenizer_path.is_file():
        raise BundleError(f"tokenizer missing: {tokenizer_path}")

    ext = engine.get("backend", {}).get("extensions")
    if ext and not (root / ext).is_file():
        raise BundleError(f"backend extension config missing: {root / ext}")

    chat_template: dict[str, str] = {}
    qairt_version: Optional[str] = None
    meta_path = root / METADATA_NAME
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            chat_template = dict(meta.get("genie", {}).get("chat_template", {}))
            qairt_version = meta.get("tool_versions", {}).get("qairt")
        except (json.JSONDecodeError, AttributeError):
            pass  # metadata is informational only

    return GenieBundle(
        model_dir=root,
        config_path=config_path,
        ctx_bins=ctx_bins,
        tokenizer_path=tokenizer_path,
        context_size=context_size,
        chat_template=chat_template,
        qairt_version=qairt_version,
    )


def find_genie_binary(explicit: str | os.PathLike[str] | None = None) -> Optional[Path]:
    """Locate ``genie-t2t-run``: explicit arg > env var > SDK root > PATH."""
    candidates: list[str] = []
    if explicit:
        candidates.append(str(explicit))
    env_bin = os.environ.get(ENV_GENIE_BIN)
    if env_bin:
        candidates.append(env_bin)
    for var in ENV_SDK_ROOTS:
        sdk = os.environ.get(var)
        if sdk:
            # TODO: verify API signature — SDK layout assumed to be
            # <sdk>/bin/<arch>/genie-t2t-run[.exe], e.g. bin/aarch64-windows-msvc/.
            candidates.extend(glob.glob(os.path.join(sdk, "bin", "*", GENIE_BIN_NAME + "*")))
    which = shutil.which(GENIE_BIN_NAME)
    if which:
        candidates.append(which)
    for c in candidates:
        p = Path(c)
        if p.is_file():
            return p
    return None


# -- the client ----------------------------------------------------------------


def _empty_result(error: Optional[str] = None) -> dict[str, Any]:
    return {
        "text": "",
        "ttft_ms": 0.0,
        "total_ms": 0.0,
        "tokens_generated": 0,
        "error": error,
    }


class InferenceClient:
    """Runs prompts against the compiled Qwen3 bundle on the NPU via Genie.

    Construction never raises for a missing model or runtime; problems are
    remembered and surfaced as ``error`` on every ``generate()`` call so a
    session can still run (the state machine degrades to fallback hints).
    """

    def __init__(
        self,
        model_dir: str | os.PathLike[str] | None = None,
        genie_bin: str | os.PathLike[str] | None = None,
        timeout_s: Optional[float] = None,
    ) -> None:
        self.bundle: Optional[GenieBundle] = None
        self.genie_bin: Optional[Path] = None
        self.setup_error: Optional[str] = None
        self.timeout_s = float(
            timeout_s if timeout_s is not None else os.environ.get(ENV_TIMEOUT_S, DEFAULT_TIMEOUT_S)
        )
        self.last_result: Optional[dict[str, Any]] = None

        try:
            self.bundle = load_bundle(model_dir)
        except BundleError as e:
            self.setup_error = str(e)
            return

        self.genie_bin = find_genie_binary(genie_bin)
        if self.genie_bin is None:
            self.setup_error = (
                f"{GENIE_BIN_NAME} not found. Set {ENV_GENIE_BIN} to the executable "
                f"or {'/'.join(ENV_SDK_ROOTS)} to the QAIRT SDK root."
            )

    # -- the one public method -------------------------------------------------

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
        if self.setup_error or self.bundle is None or self.genie_bin is None:
            result = _empty_result(self.setup_error or "client not initialised")
            self.last_result = result
            return result
        if not prompt.strip():
            result = _empty_result("empty prompt")
            self.last_result = result
            return result

        try:
            result = self._run_genie(prompt, max_tokens)
        except Exception as e:  # noqa: BLE001 - graceful degradation is the contract
            result = _empty_result(f"{type(e).__name__}: {e}")
        self.last_result = result
        return result

    # -- Genie transport -------------------------------------------------------

    def _run_genie(self, prompt: str, max_tokens: int) -> dict[str, Any]:
        assert self.bundle is not None and self.genie_bin is not None
        bundle = self.bundle

        with tempfile.TemporaryDirectory(prefix="socratic_genie_") as tmp:
            tmp_dir = Path(tmp)
            prompt_file = tmp_dir / "prompt.txt"
            profile_file = tmp_dir / "profile.json"
            prompt_file.write_text(prompt, encoding="utf-8")

            config_path = self._apply_max_tokens(bundle.config_path, max_tokens, tmp_dir)

            # Source for flags: qai_hub_models/utils/llm/genie/device_scripts/run_android.py
            #   genie-t2t-run -c genie_config.json --prompt_file <file> --profile <json>
            cmd = [
                str(self.genie_bin),
                "-c",
                str(config_path),
                "--prompt_file",
                str(prompt_file),
                "--profile",
                str(profile_file),
            ]

            # TODO: verify API signature — on Windows the QAIRT runtime DLLs
            # (Genie.dll, QnnHtp*.dll, QnnSystem.dll) and the hexagon-v73 skel
            # must be discoverable; the SDK docs put <sdk>/lib/<arch> and
            # <sdk>/lib/hexagon-v73/unsigned on PATH. Nothing is added here.
            env = os.environ.copy()

            t_start = time.perf_counter()
            proc = subprocess.Popen(
                cmd,
                cwd=str(bundle.model_dir),  # config paths are relative to the bundle
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )

            stdout_chunks: list[bytes] = []
            first_output_at: list[float] = []  # wall-clock fallback for TTFT

            def _pump() -> None:
                assert proc.stdout is not None
                seen_begin = False
                buf = b""
                while True:
                    chunk = proc.stdout.read(64)
                    if not chunk:
                        break
                    stdout_chunks.append(chunk)
                    if not seen_begin:
                        buf += chunk
                        idx = buf.find(BEGIN_MARKER.encode())
                        if idx != -1 and len(buf) > idx + len(BEGIN_MARKER):
                            first_output_at.append(time.perf_counter())
                            seen_begin = True

            reader = threading.Thread(target=_pump, daemon=True)
            reader.start()
            try:
                proc.wait(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                return _empty_result(f"genie-t2t-run timed out after {self.timeout_s:.0f}s")
            reader.join(timeout=5)
            t_end = time.perf_counter()

            stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace")
            stderr = (proc.stderr.read() if proc.stderr else b"").decode("utf-8", errors="replace")

            if proc.returncode != 0:
                tail = stderr.strip().splitlines()[-3:] if stderr.strip() else []
                return _empty_result(
                    f"genie-t2t-run exited {proc.returncode}: {' | '.join(tail) or 'no stderr'}"
                )

            text = self._parse_output(stdout)
            if text is None:
                return _empty_result("no [BEGIN]/[END] markers in genie-t2t-run output")

            total_ms = (t_end - t_start) * 1000.0
            wall_ttft_ms = (
                (first_output_at[0] - t_start) * 1000.0 if first_output_at else total_ms
            )
            profile = self._parse_profile(profile_file)

            result = _empty_result(None)
            result["text"] = text
            result["total_ms"] = round(total_ms, 1)
            # Prefer Genie's own TTFT (excludes model load); fall back to wall clock.
            result["ttft_ms"] = round(profile.get("ttft_ms", wall_ttft_ms), 1)
            result["tokens_generated"] = self._count_tokens(text, profile)
            # Extra diagnostics, harmless to callers that only read the 5 keys.
            result["latency_source"] = "genie_profile" if "ttft_ms" in profile else "wall_clock"
            result["tokens_per_second"] = profile.get("tokens_per_second")
            result["max_tokens_requested"] = max_tokens
            return result

    @staticmethod
    def _apply_max_tokens(config_path: Path, max_tokens: int, tmp_dir: Path) -> Path:
        """Return the config file to run with, honouring ``max_tokens``.

        # TODO: verify API signature — the Genie dialog config key that caps
        # generated tokens is not confirmed. Until it is, the original
        # genie_config.json is used unchanged and max_tokens is only recorded
        # in the result. Once confirmed: copy the config into ``tmp_dir``
        # with the cap applied and return that path (paths inside the config
        # are relative to the bundle dir, which stays the cwd).
        """
        return config_path

    @staticmethod
    def _parse_output(stdout: str) -> Optional[str]:
        """Completion text between ``[BEGIN]:`` and ``[END]`` (see BEGIN_MARKER)."""
        begin = stdout.find(BEGIN_MARKER)
        if begin == -1:
            return None
        text = stdout[begin + len(BEGIN_MARKER):]
        end = text.find(END_MARKER)
        if end != -1:
            text = text[:end]
        return text.strip()

    @staticmethod
    def _parse_profile(profile_file: Path) -> dict[str, float]:
        """Read Genie's ``--profile`` JSON.

        Layout and units are taken from
        ``qai_hub_models/utils/llm/genie/jobs.py::compute_genie_metrics``:
        ``components[0]["events"][1]`` holds ``time-to-first-token`` (µs),
        ``token-generation-rate`` (tokens/s) and ``prompt-processing-rate``.
        Missing file or unexpected shape -> empty dict (callers fall back).
        """
        out: dict[str, float] = {}
        if not profile_file.is_file():
            return out
        try:
            data = json.loads(profile_file.read_text(encoding="utf-8"))
            event = data["components"][0]["events"][1]
            out["ttft_ms"] = float(event["time-to-first-token"]["value"]) / 1000.0
            out["tokens_per_second"] = float(event["token-generation-rate"]["value"])
            out["prefill_tokens_per_second"] = float(event["prompt-processing-rate"]["value"])
            # TODO: verify API signature — if the profile also reports the
            # generated-token count, read it here under "num_generated_tokens".
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            return {}
        return out

    def _count_tokens(self, text: str, profile: dict[str, float]) -> int:
        """Generated-token count: profile > local tokenizer > word estimate."""
        if "num_generated_tokens" in profile:
            return int(profile["num_generated_tokens"])
        if self.bundle is not None:
            try:
                from tokenizers import Tokenizer  # optional; not a hard dependency

                tok = Tokenizer.from_file(str(self.bundle.tokenizer_path))
                return len(tok.encode(text, add_special_tokens=False).ids)
            except Exception:  # noqa: BLE001 - library absent or file unreadable
                pass
        # Rough estimate only (whitespace words); real counts come from the two paths above.
        return len(text.split())
