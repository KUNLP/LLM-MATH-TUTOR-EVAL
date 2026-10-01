"""OpenAI Responses inference and resumable Batch inference, without GPU imports.

The SDK is optional until a real client is constructed. Batch checkpoints contain
job identifiers, never API keys or conversation payloads. A timed-out job is left
running and the same payload resumes it on the next call.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
import time
from collections.abc import Collection, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from pathlib import Path
from typing import Any

from .util_for_types import GenerationRequest, GenerationResult, ModelSpec

logger = logging.getLogger(__name__)

_CONFIGURATION_PARAMETERS = frozenset(
    {"model", "temperature", "top_p", "max_tokens", "max_output_tokens", "n", "seed",
     "instructions", "stream", "background"}
)


class APIConfigurationError(ValueError):
    """The API client or request configuration needs correction."""


class APIAuthenticationError(RuntimeError):
    """Authentication/permission failures should stop the whole run."""


class BatchTimeoutError(TimeoutError):
    """A saved remote batch is still pending; rerun to resume it."""


class BatchProtocolError(RuntimeError):
    """A malformed batch cannot safely be assigned to the requested samples."""


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _error_code(value: Any) -> str:
    code = str(_get(value, "code", "") or "").lower()
    message = str(_get(value, "message", "") or "").lower()
    if (
        "context_length" in code
        or "context_window" in code
        or any(
            phrase in message for phrase in ("context length", "context window", "too many tokens")
        )
    ):
        return "context_length"
    return "request_error"


def _failure(request_id: str, message: str, code: str = "request_error") -> GenerationResult:
    return GenerationResult(request_id=request_id, error=message, error_code=code)


def _check_global_error(
    status: Any, error: Any, *, configuration_parameters: Collection[str] = ()
) -> None:
    code = _get(error, "code")
    if status in (401, 403) or code in {"invalid_api_key", "insufficient_permissions"}:
        raise APIAuthenticationError(
            "API authentication or permission failed; check the configured credential."
        )
    if code == "insufficient_quota":
        raise APIConfigurationError(
            "The API account has insufficient quota; check account limits before resuming."
        )
    # Even an invalid_request_error may describe just one overlong dialogue.
    if _error_code(error) == "context_length":
        return
    if code in {
        "model_not_found",
        "unsupported_parameter",
        "unsupported_value",
        "invalid_model",
        "invalid_parameter",
        "invalid_request",
        "invalid_request_error",
    }:
        raise APIConfigurationError("The API rejected the model or generation configuration.")
    # Some API errors omit code, leaving only type/param. Check parameters
    # shared by the run, including nested options such as reasoning.effort,
    # without turning content-specific input failures into global failures.
    parameter = _get(error, "param")
    root = parameter.split(".", 1)[0].split("[", 1)[0] if isinstance(parameter, str) else None
    configured = _CONFIGURATION_PARAMETERS.union(configuration_parameters)
    if (
        not code
        and status in (None, 400, 422)
        and (status in (400, 422) or _get(error, "type") == "invalid_request_error")
        and root in configured
    ):
        raise APIConfigurationError("The API rejected the model or generation configuration.")


def parse_response(
    request_id: str, response: Any, *, configuration_parameters: Collection[str] = ()
) -> GenerationResult:
    """Read every message/output_text block, never reasoning or tool output."""
    error = _get(response, "error")
    if error:
        _check_global_error(None, error, configuration_parameters=configuration_parameters)
        return _failure(request_id, "The API returned an error response.", _error_code(error))
    status = _get(response, "status")
    if status in {"failed", "cancelled", "queued", "in_progress"}:
        return _failure(request_id, f"The API response has no final answer (status={status}).")
    output = _get(response, "output")
    if not isinstance(output, (list, tuple)):
        return _failure(request_id, "Malformed API response: output must be a list.")
    pieces: list[str] = []
    for item in output:
        if _get(item, "type") != "message":
            continue
        content = _get(item, "content")
        if not isinstance(content, (list, tuple)):
            return _failure(request_id, "Malformed API response: message content must be a list.")
        for part in content:
            kind = _get(part, "type")
            if kind == "refusal":
                return _failure(request_id, "The API refused to provide an answer.")
            if kind == "output_text":
                value = _get(part, "text")
                if not isinstance(value, str):
                    return _failure(
                        request_id, "Malformed API response: output_text is not a string."
                    )
                pieces.append(value)
    text = "\n".join(pieces).strip()
    if not text:
        return _failure(request_id, "The API returned no answer text.")
    reason = _get(_get(response, "incomplete_details"), "reason")
    return GenerationResult(
        request_id=request_id, text=text, finish_reason=reason or status or "completed"
    )


def _positive_number(options: dict, name: str, default: float) -> float:
    value = options.get(name, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise APIConfigurationError(f"{name} must be a finite positive number.")
    return float(value)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class APIBackend:
    def __init__(self, spec: ModelSpec, work_dir: Path, *, client: Any = None):
        self.spec = spec
        self.work_dir = Path(work_dir)
        options = spec.options
        self.mode = options.get("mode", "responses")
        if self.mode not in {"responses", "batch"}:
            raise APIConfigurationError("API mode must be 'responses' or 'batch'.")
        if not isinstance(spec.model, str) or not spec.model.strip():
            raise APIConfigurationError("An API model identifier is required.")
        self.max_concurrency = options.get("max_concurrency", 8)
        if type(self.max_concurrency) is not int or self.max_concurrency < 1:
            raise APIConfigurationError("max_concurrency must be a positive integer.")
        self.timeout = _positive_number(options, "timeout", 120)
        self.poll_interval = _positive_number(options, "poll_interval", 10)
        self.batch_timeout = _positive_number(options, "batch_timeout", 90000)
        self.max_retries = options.get("max_retries", 3)
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise APIConfigurationError("max_retries must be a nonnegative integer.")
        self.request_options = deepcopy(options.get("request_options", {}))
        if not isinstance(self.request_options, dict):
            raise APIConfigurationError("request_options must be a mapping.")
        protected = {
            "model",
            "input",
            "instructions",
            "max_tokens",
            "max_output_tokens",
            "stream",
            "background",
            "extra_body",
            "temperature",
            "top_p",
            "seed",
            "n",
        }
        if protected.intersection(self.request_options):
            raise APIConfigurationError(
                "request_options cannot override model, input, instructions, token limits, or execution mode."
            )
        if set(spec.generation) - {"max_tokens", "temperature", "top_p", "n"}:
            raise APIConfigurationError(
                "API generation supports only max_tokens, temperature, top_p, and n=1."
            )
        if type(spec.generation.get("n", 1)) is not int or spec.generation.get("n", 1) != 1:
            raise APIConfigurationError("The API backend requires generation.n=1.")
        self.max_tokens = spec.generation.get("max_tokens", 1000)
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise APIConfigurationError("generation.max_tokens must be a positive integer.")
        # An injected client makes all unit tests independent of the SDK/network.
        self.client = client if client is not None else self._make_client()

    def _make_client(self) -> Any:
        options = self.spec.options
        key_env = options.get("api_key_env", "OPENAI_API_KEY")
        if not isinstance(key_env, str) or not key_env:
            raise APIConfigurationError("api_key_env must be a nonempty environment variable name.")
        key = os.environ.get(key_env, "").strip()
        if not key and options.get("api_key_file"):
            try:
                key = Path(options["api_key_file"]).expanduser().read_text(encoding="utf-8").strip()
            except (OSError, TypeError):
                raise APIConfigurationError(
                    "The configured API key file could not be read."
                ) from None
        if not key:
            raise APIConfigurationError(
                "No API key is configured; set api_key_env or api_key_file."
            )
        try:
            from openai import OpenAI
        except ImportError:
            raise APIConfigurationError(
                "Install the optional openai dependency to use the API backend."
            ) from None
        kwargs = {"api_key": key, "timeout": self.timeout, "max_retries": self.max_retries}
        if options.get("base_url"):
            kwargs["base_url"] = options["base_url"]
        return OpenAI(**kwargs)

    def _payload(self, request: GenerationRequest) -> dict:
        if not isinstance(request.system_prompt, str) or not isinstance(request.messages, list):
            raise APIConfigurationError(
                "Requests need a string system_prompt and a list of messages."
            )
        for message in request.messages:
            if (
                not isinstance(message, dict)
                or message.get("role") not in {"user", "assistant", "system", "developer"}
                or not isinstance(message.get("content"), str)
            ):
                raise APIConfigurationError("Each message needs an OpenAI role and string content.")
        payload = deepcopy(self.request_options)
        payload.update(
            model=self.spec.model,
            input=deepcopy(request.messages),
            instructions=request.system_prompt,
            max_output_tokens=self.max_tokens,
        )
        for key in ("temperature", "top_p"):
            value = self.spec.generation.get(key)
            if value is not None:
                payload[key] = value
        return payload

    def _request(self, request: GenerationRequest, payload: dict) -> GenerationResult:
        try:
            response = self.client.responses.create(**payload)
        except Exception as exc:
            body = _get(exc, "body", {})
            error = _get(body, "error", body) or {"code": _get(exc, "code"), "message": str(exc)}
            _check_global_error(
                _get(exc, "status_code"), error, configuration_parameters=self.request_options
            )
            if type(exc).__name__ in {"AuthenticationError", "PermissionDeniedError"}:
                raise APIAuthenticationError("API authentication or permission failed.") from None
            if isinstance(exc, (TypeError, ValueError)):
                raise APIConfigurationError(
                    "The API client rejected the request configuration."
                ) from None
            # Exception text may contain credentials or prompts; keep persisted errors generic.
            return _failure(
                request.request_id,
                "API request failed after the configured SDK retries.",
                _error_code(error),
            )
        return parse_response(
            request.request_id, response, configuration_parameters=self.request_options
        )

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        requests = list(requests)
        ids = [request.request_id for request in requests]
        if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(
            ids
        ):
            raise APIConfigurationError(
                "Request IDs must be nonempty and unique within a generation call."
            )
        if not requests:
            return []
        payloads = [self._payload(request) for request in requests]
        if self.mode == "batch":
            return self._batch(requests, payloads)
        # Keep only max_concurrency futures queued, so a global failure does not
        # launch the rest of a large evaluation before it can be reported.
        results: dict[str, GenerationResult] = {}
        iterator = iter(zip(requests, payloads))
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            pending = {}
            for _ in range(min(self.max_concurrency, len(requests))):
                request, payload = next(iterator)
                pending[executor.submit(self._request, request, payload)] = request.request_id
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                try:
                    for future in done:
                        request_id = pending.pop(future)
                        result = future.result()
                        if result.request_id != request_id:
                            raise BatchProtocolError("API response request ID mismatch.")
                        results[request_id] = result
                except BaseException:
                    for future in pending:
                        future.cancel()
                    raise
                for _ in done:
                    entry = next(iterator, None)
                    if entry is None:
                        break
                    request, payload = entry
                    pending[executor.submit(self._request, request, payload)] = request.request_id
        return [results[request_id] for request_id in ids]

    def _batch(
        self, requests: list[GenerationRequest], payloads: list[dict]
    ) -> list[GenerationResult]:
        rows = [
            {"custom_id": req.request_id, "method": "POST", "url": "/v1/responses", "body": payload}
            for req, payload in zip(requests, payloads)
        ]
        # Canonical sorting allows resumption even when the caller changes order.
        rows.sort(key=lambda row: row["custom_id"])
        data = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ).encode("utf-8")
        identity = (
            str(self.spec.options.get("base_url", "https://api.openai.com/v1")).encode("utf-8")
            + b"\n"
            + data
        )
        digest = hashlib.sha256(identity).hexdigest()
        checkpoint = self.work_dir / "api_batches" / (digest + ".json")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        # flock prevents two processes from submitting the same saved payload.
        import fcntl

        with checkpoint.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                return self._batch_locked(requests, data, digest, checkpoint)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _batch_locked(
        self, requests: list[GenerationRequest], data: bytes, digest: str, checkpoint: Path
    ) -> list[GenerationResult]:
        if checkpoint.exists():
            try:
                state = json.loads(checkpoint.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                raise BatchProtocolError(
                    "The saved batch checkpoint is unreadable; refusing to resubmit."
                ) from None
            if not isinstance(state, dict) or state.get("payload_hash") != digest:
                raise BatchProtocolError("The saved batch checkpoint does not match its payload.")
        else:
            state = {"version": 1, "payload_hash": digest, "status": "new"}
        if not state.get("batch_id"):
            if state.get("status") == "submitting":
                raise BatchProtocolError(
                    "A previous batch submission has an unknown outcome. Recover its batch_id before retrying to avoid duplicate charges."
                )
            if not state.get("input_file_id"):
                uploaded = self.client.files.create(
                    file=("requests.jsonl", data, "application/jsonl"), purpose="batch"
                )
                file_id = _get(uploaded, "id")
                if not isinstance(file_id, str) or not file_id:
                    raise BatchProtocolError("Batch upload returned no file ID.")
                state.update(input_file_id=file_id, status="uploaded")
                _atomic_json(checkpoint, state)
            state["status"] = "submitting"
            _atomic_json(checkpoint, state)
            # A lost response can hide an accepted job. Disable SDK retries
            # only for creation; the original client still retries reads and
            # ordinary inference. The clone shares its transport, so do not
            # close it independently of self.client.
            job = self.client.with_options(max_retries=0).batches.create(
                input_file_id=state["input_file_id"],
                endpoint="/v1/responses",
                completion_window="24h",
            )
            batch_id = _get(job, "id")
            if not isinstance(batch_id, str) or not batch_id:
                raise BatchProtocolError(
                    "Batch submission returned no job ID; refusing automatic resubmission."
                )
            state.update(batch_id=batch_id, status=_get(job, "status", "validating"))
            _atomic_json(checkpoint, state)
            logger.info("Submitted API batch %s (%d requests)", batch_id, len(requests))
        else:
            logger.info("Resuming API batch %s", state["batch_id"])
        deadline = time.monotonic() + self.batch_timeout
        terminal = {"completed", "failed", "expired", "cancelled"}
        pending_statuses = {"validating", "in_progress", "finalizing", "cancelling"}
        last_status = None
        while True:
            job = self.client.batches.retrieve(state["batch_id"])
            status = _get(job, "status")
            if status not in terminal | pending_statuses:
                raise BatchProtocolError("Batch retrieval returned an unknown status.")
            state["status"] = status
            if status != last_status:
                logger.info("API batch %s: %s", state["batch_id"], status)
                last_status = status
            _atomic_json(checkpoint, state)
            if status in terminal:
                errors = _get(_get(job, "errors"), "data", []) or []
                for error in errors:
                    _check_global_error(
                        None, error, configuration_parameters=self.request_options
                    )
                return self._collect_batch(requests, job)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BatchTimeoutError(
                    f"Batch {state['batch_id']} is still pending; its ID is saved for resumption."
                )
            time.sleep(min(self.poll_interval, remaining))

    def _collect_batch(self, requests: list[GenerationRequest], job: Any) -> list[GenerationResult]:
        expected = {request.request_id for request in requests}
        found: dict[str, GenerationResult] = {}
        file_ids = [_get(job, field) for field in ("output_file_id", "error_file_id")]
        for file_id in dict.fromkeys(value for value in file_ids if value):
            content = self.client.files.content(file_id)
            text = _get(content, "text")
            if not isinstance(text, str):
                raw = content if isinstance(content, (str, bytes)) else content.read()
                text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if not isinstance(text, str):
                raise BatchProtocolError("Batch result file is not text.")
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    raise BatchProtocolError("Batch result file contains malformed JSON.") from None
                if not isinstance(row, dict):
                    raise BatchProtocolError("A batch result row is not an object.")
                request_id = row.get("custom_id")
                if (
                    not isinstance(request_id, str)
                    or request_id not in expected
                    or request_id in found
                ):
                    raise BatchProtocolError(
                        "Batch result contains an unknown, missing, or duplicate custom_id."
                    )
                error = row.get("error")
                response = row.get("response")
                if error:
                    _check_global_error(
                        None, error, configuration_parameters=self.request_options
                    )
                    result = _failure(request_id, "The batch request failed.", _error_code(error))
                elif isinstance(response, dict):
                    status = response.get("status_code")
                    body = response.get("body")
                    if type(status) is not int or not isinstance(body, dict):
                        raise BatchProtocolError(
                            "Batch response has a malformed HTTP status or body."
                        )
                    if not 200 <= status < 300:
                        error = body.get("error", body)
                        _check_global_error(
                            status, error, configuration_parameters=self.request_options
                        )
                        result = _failure(
                            request_id,
                            "The batch request returned an HTTP error.",
                            _error_code(error),
                        )
                    else:
                        result = parse_response(
                            request_id, body, configuration_parameters=self.request_options
                        )
                else:
                    raise BatchProtocolError("Batch result has neither a response nor an error.")
                found[request_id] = result
        status = _get(job, "status")
        for request_id in expected - found.keys():
            found[request_id] = _failure(
                request_id, f"Batch {status} without a result for this request."
            )
        return [found[request.request_id] for request in requests]

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close is not None:
            close()
