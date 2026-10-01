"""vLLM inference in an owned, spawned process; importing this module needs no GPU."""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
import signal
import time
from collections.abc import Sequence
from multiprocessing.connection import Connection
from threading import Event
from typing import Any

from .util_for_types import GenerationRequest, GenerationResult, ModelSpec

logger = logging.getLogger(__name__)
_STARTUP_LOG_INTERVAL = 60.0


class VLLMBackendError(RuntimeError):
    """A worker failed, exited, or exceeded its configured time limit."""


def _validate_spec(spec: ModelSpec) -> None:
    if spec.generation.get("n", 1) != 1:
        raise ValueError("vLLM generation.n must be 1")
    max_tokens = spec.generation.get("max_tokens", 1000)
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("vLLM generation.max_tokens must be a positive integer")
    if spec.options.get("system_prompt_mode", "auto") not in {"auto", "system", "user"}:
        raise ValueError("system_prompt_mode must be auto, system, or user")
    engine_kwargs = spec.options.get("engine_kwargs", {})
    if not isinstance(engine_kwargs, dict):
        raise ValueError("engine_kwargs must be a mapping")
    if {"model", "tokenizer"} & engine_kwargs.keys():
        raise ValueError("engine_kwargs cannot override model or tokenizer")
    for key, default in (
        ("startup_timeout", 600),
        ("request_timeout", 600),
        ("shutdown_timeout", 30),
    ):
        timeout = float(spec.options.get(key, default))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(f"{key} must be finite and positive")


def _messages(
    request: GenerationRequest, *, system_role: bool, leading_user: bool = False
) -> list[dict[str, str]]:
    messages = [dict(message) for message in request.messages]
    if not messages:
        raise ValueError("At least one chat message is required")
    if any(message.get("role") not in {"user", "assistant"} for message in messages):
        raise ValueError("Conversation roles must be user or assistant")
    # Preserve teacher histories beginning with their own problem statement.
    # Only a template rejection enables the empty role adapter below.
    if leading_user and messages[0]["role"] == "assistant":
        messages.insert(0, {"role": "user", "content": ""})
    if request.system_prompt:
        if system_role:
            messages.insert(0, {"role": "system", "content": request.system_prompt})
        else:
            first_user = next((message for message in messages if message["role"] == "user"), None)
            if first_user is None:
                raise ValueError("A user message is required to fold the system instruction")
            first_user["content"] = request.system_prompt + "\n\n" + first_user["content"]
    return messages


def _render_chat_tokens(
    tokenizer: Any,
    requests: Sequence[GenerationRequest],
    mode: str,
    *,
    add_generation_prompt: bool,
    continue_final_message: bool = False,
) -> list[list[int]]:
    """Render related prompts under the same template fallback policy.

    In particular, answer-scoring prefixes and full continuations must select
    the same system-role and leading-user treatment before their tokens differ.
    """
    if mode not in {"auto", "system", "user"}:
        raise ValueError("system_prompt_mode must be auto, system, or user")
    system_modes = (True, False) if mode == "auto" else (mode == "system",)
    needs_leading_user = any(
        request.messages and request.messages[0].get("role") == "assistant"
        for request in requests
    )
    leading_modes = (False, True) if needs_leading_user else (False,)
    last_error = None
    for system_role in system_modes:
        for leading_user in leading_modes:
            try:
                rendered = []
                for request in requests:
                    options = {
                        "tokenize": True,
                        "add_generation_prompt": add_generation_prompt,
                        # Transformers 5 otherwise defaults to BatchEncoding.
                        "return_dict": False,
                    }
                    if continue_final_message:
                        options["continue_final_message"] = True
                    tokens = tokenizer.apply_chat_template(
                        _messages(request, system_role=system_role, leading_user=leading_user),
                        **options,
                    )
                    rendered.append(list(tokens))
                return rendered
            except Exception as exc:
                last_error = exc
    # No history is dropped or truncated when all supported representations fail.
    assert last_error is not None
    raise last_error


def _render_tokens(tokenizer: Any, request: GenerationRequest, mode: str) -> list[int]:
    return _render_chat_tokens(
        tokenizer, [request], mode, add_generation_prompt=True
    )[0]


def _generate(
    llm: Any, sampling_type: Any, spec: ModelSpec, requests: Sequence[GenerationRequest]
) -> list[GenerationResult]:
    tokenizer = llm.get_tokenizer()
    max_model_len = llm.llm_engine.model_config.max_model_len
    generation = dict(spec.generation)
    generation.setdefault("temperature", 0.0)
    generation.setdefault("max_tokens", 1000)
    generation["n"] = 1
    max_tokens = generation["max_tokens"]
    mode = spec.options.get("system_prompt_mode", "auto")
    results: list[GenerationResult | None] = [None] * len(requests)
    safe_indices, prompts, sampling = [], [], []
    for index, request in enumerate(requests):
        tokens = _render_tokens(tokenizer, request, mode)
        if len(tokens) + max_tokens > max_model_len:
            results[index] = GenerationResult(
                request_id=request.request_id,
                error=f"Prompt ({len(tokens)} tokens) plus generation budget ({max_tokens}) exceeds context ({max_model_len})",
                error_code="context_length",
            )
            continue
        options = dict(generation)
        options["seed"] = request.seed
        sampling.append(sampling_type(**options))
        prompts.append({"prompt_token_ids": tokens})
        safe_indices.append(index)
    if prompts:
        outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
        if len(outputs) != len(safe_indices):
            raise VLLMBackendError("vLLM returned a different number of outputs than requests")
        for index, output in zip(safe_indices, outputs):
            if len(output.outputs) != 1:
                raise VLLMBackendError("vLLM must return exactly one completion per request")
            completion = output.outputs[0]
            results[index] = GenerationResult(
                request_id=requests[index].request_id,
                text=completion.text,
                finish_reason=completion.finish_reason,
            )
    if any(result is None for result in results):
        raise VLLMBackendError("Incomplete vLLM result mapping")
    return results  # type: ignore[return-value]


def _configure_environment(spec: ModelSpec) -> None:
    options = spec.options
    if options.get("cuda_visible_devices") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(options["cuda_visible_devices"])
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    if options.get("cache_dir"):
        os.environ["HF_HOME"] = str(options["cache_dir"])
        os.environ["HF_HUB_CACHE"] = str(options["cache_dir"])
    if "offline" in options:
        offline = "1" if options["offline"] else "0"
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[name] = offline
    token_env = options.get("hf_token_env", "HF_TOKEN")
    token_file = options.get("hf_token_file")
    if os.environ.get(token_env):
        os.environ["HF_TOKEN"] = os.environ[token_env]
    elif not os.environ.get("HF_TOKEN") and token_file:
        with open(token_file, encoding="utf-8") as handle:
            os.environ["HF_TOKEN"] = handle.read().strip()


def _worker(connection: Connection, spec: ModelSpec) -> None:
    llm = None
    try:
        if os.name == "posix":
            os.setsid()
        connection.send(("started", os.getpid()))
        _configure_environment(spec)
        connection.send(("startup", "importing vLLM"))
        # Do not move these imports to the parent: CUDA must initialize only in
        # the owned spawn process, after device and cache configuration.
        from vllm import LLM, SamplingParams

        kwargs = {"dtype": "auto", "tensor_parallel_size": 1, "gpu_memory_utilization": 0.9}
        kwargs.update(spec.options.get("engine_kwargs", {}))
        if spec.options.get("cache_dir"):
            kwargs.setdefault("download_dir", spec.options["cache_dir"])
        connection.send(("startup", "preparing model (checkpoint download/load and engine warmup)"))
        llm = LLM(model=spec.model, tokenizer=spec.model, **kwargs)
        connection.send(("ready", None))
        while True:
            operation, payload = connection.recv()
            if operation == "close":
                break
            if operation != "generate":
                raise ValueError("Unknown vLLM worker operation")
            connection.send(("result", _generate(llm, SamplingParams, spec, payload)))
    except EOFError:
        pass
    except BaseException as exc:
        try:
            connection.send(("error", f"{type(exc).__name__}: {exc}"))
        except (EOFError, OSError):
            pass
    finally:
        if llm is not None:
            # Public close methods differ by vLLM version. The process group is
            # the final resource boundary, even when engine shutdown hangs.
            try:
                shutdown = getattr(llm, "shutdown", None)
                if callable(shutdown):
                    shutdown()
                else:
                    engine = getattr(llm, "llm_engine", None)
                    executor = getattr(engine, "model_executor", None)
                    shutdown = getattr(executor, "shutdown", None)
                    if callable(shutdown):
                        shutdown()
            except Exception:
                pass
        connection.close()


class VLLMBackend:
    """One local model whose CUDA lifetime ends before ``close`` returns.

    This backend is intentionally synchronous. Callers must serialize generate
    and close operations. The owning DP pool keeps each replica resident and
    calls independent replicas concurrently.
    """

    def __init__(self, spec: ModelSpec, *, worker_target=None):
        _validate_spec(spec)
        self.spec = spec
        self._closed = False
        self._group_owned = False
        self._startup_stage = "starting worker process"
        self._cancel_requested = Event()
        self._connection, child = multiprocessing.get_context("spawn").Pipe()
        self._process = multiprocessing.get_context("spawn").Process(
            target=worker_target or _worker,
            args=(child, spec),
            name=f"vllm-{spec.name}",
            daemon=False,
        )
        try:
            logger.info("Starting vLLM model %s on GPU(s) %s; startup timeout %ss includes "
                        "checkpoint downloads, weight loading and engine warmup",
                        spec.model, spec.options.get("cuda_visible_devices", "inherited"),
                        spec.options.get("startup_timeout", 600))
            self._process.start()
            child.close()
            self._receive("ready", float(spec.options.get("startup_timeout", 600)))
            logger.info("vLLM model %s is ready on GPU(s) %s", spec.model,
                        spec.options.get("cuda_visible_devices", "inherited"))
        except BaseException:
            child.close()
            self.close()
            raise

    def _receive(self, expected: str, timeout: float) -> Any:
        started = time.monotonic()
        deadline = started + timeout
        next_startup_log = started + _STARTUP_LOG_INTERVAL
        while True:
            if self._cancel_requested.is_set():
                raise VLLMBackendError("vLLM worker request was cancelled")
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                detail = (f"vLLM worker timed out after {timeout:g}s while waiting for {expected}: "
                          f"model={self.spec.model}, "
                          f"GPU(s)={self.spec.options.get('cuda_visible_devices', 'inherited')}")
                if expected == "ready":
                    detail += (f"; last startup stage: {self._startup_stage}. "
                               "Checkpoint downloads count toward startup_timeout; "
                               "increase startup_timeout for an uncached model.")
                raise VLLMBackendError(detail)
            if expected == "ready" and now >= next_startup_log:
                logger.info("Still preparing %s on GPU(s) %s: %s; %.0fs elapsed, %.0fs remaining",
                            self.spec.model, self.spec.options.get("cuda_visible_devices", "inherited"),
                            self._startup_stage, now - started, remaining)
                next_startup_log = now + _STARTUP_LOG_INTERVAL
            if self._connection.poll(min(remaining, 0.1)):
                try:
                    kind, payload = self._connection.recv()
                except (EOFError, OSError) as exc:
                    raise VLLMBackendError(
                        "vLLM worker closed its connection unexpectedly"
                    ) from exc
                if kind == "started":
                    if payload != self._process.pid:
                        raise VLLMBackendError("Unexpected vLLM worker process identity")
                    self._group_owned = os.name == "posix"
                    continue
                if kind == "startup" and expected == "ready" and isinstance(payload, str):
                    self._startup_stage = payload
                    logger.info("vLLM %s on GPU(s) %s: %s", self.spec.model,
                                self.spec.options.get("cuda_visible_devices", "inherited"), payload)
                    continue
                if kind == "error":
                    raise VLLMBackendError(f"vLLM worker failed: {payload}")
                if kind != expected:
                    raise VLLMBackendError(f"Unexpected worker reply: {kind}")
                return payload
            if not self._process.is_alive():
                raise VLLMBackendError(
                    f"vLLM worker exited unexpectedly (exit code {self._process.exitcode})"
                )

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        if self._closed:
            raise VLLMBackendError("vLLM backend is closed")
        if not requests:
            return []
        try:
            self._connection.send(("generate", list(requests)))
            results = self._receive("result", float(self.spec.options.get("request_timeout", 600)))
            if len(results) != len(requests) or any(
                result.request_id != request.request_id
                for result, request in zip(results, requests)
            ):
                raise VLLMBackendError("vLLM worker result IDs or order do not match requests")
            return results
        except BaseException:
            self.close()
            raise

    def cancel_pending(self) -> None:
        """Ask the active inference thread to stop; never touch its pipe here."""
        self._cancel_requested.set()

    def _signal_owned(self, sig: int) -> None:
        if self._process.pid is None:
            return
        if os.name == "posix":
            # Also covers a timeout between setsid() and the started handshake.
            try:
                own_group = self._group_owned or os.getpgid(self._process.pid) == self._process.pid
            except ProcessLookupError:
                own_group = self._group_owned
            if own_group:
                self._group_owned = True
                try:
                    os.killpg(self._process.pid, sig)
                except ProcessLookupError:
                    pass
                return
        if self._process.is_alive():
            if sig == signal.SIGTERM:
                self._process.terminate()
            else:
                self._process.kill()

    def _owned_group_alive(self) -> bool:
        if not self._group_owned or os.name != "posix":
            return False
        if os.path.isdir("/proc"):
            # killpg(..., 0) also reports orphan zombies. They hold no GPU
            # resources; inspect only live members of this owned process group.
            with os.scandir("/proc") as processes:
                for entry in processes:
                    if not entry.name.isdigit():
                        continue
                    try:
                        with open(f"/proc/{entry.name}/stat", encoding="utf-8") as handle:
                            fields = handle.read().rsplit(")", 1)[1].split()
                        if int(fields[2]) == self._process.pid and fields[0] not in {"Z", "X"}:
                            return True
                    except (OSError, ValueError, IndexError):
                        continue
            return False
        try:
            os.killpg(self._process.pid, 0)
            return True
        except ProcessLookupError:
            return False

    def _wait_for_shutdown(self, deadline: float) -> bool:
        while True:
            self._process.join(0)
            if not self._process.is_alive() and not self._owned_group_alive():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.02, remaining))

    def close(self) -> None:
        if self._closed:
            return
        timeout = float(self.spec.options.get("shutdown_timeout", 30))
        deadline = time.monotonic() + timeout
        try:
            if self._process.pid is not None:
                if self._cancel_requested.is_set():
                    # A cancelled worker may still be generating, so skip its
                    # graceful command queue and terminate its owned processes.
                    self._signal_owned(signal.SIGTERM)
                elif self._process.is_alive():
                    try:
                        if not self._connection.closed:
                            self._connection.send(("close", None))
                    except (BrokenPipeError, EOFError, OSError, ValueError):
                        pass
                stopped = self._wait_for_shutdown(deadline - timeout * 0.4)
                if not stopped:
                    # Terminate surviving descendants even if their owner exited.
                    self._signal_owned(signal.SIGTERM)
                    stopped = self._wait_for_shutdown(deadline - timeout * 0.2)
                if not stopped:
                    self._signal_owned(signal.SIGKILL)
                    stopped = self._wait_for_shutdown(deadline)
                if not stopped:
                    raise VLLMBackendError(
                        "vLLM worker group could not be reaped within shutdown_timeout"
                    )
            self._closed = True
        finally:
            self._connection.close()

    def __enter__(self) -> VLLMBackend:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
