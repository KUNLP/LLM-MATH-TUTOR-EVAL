"""Continuation and answer scoring in the simulator's owned vLLM workers.

This module imports no model runtime in the parent process. Evaluation requests
also work with ``DataParallelBackend(spec, factory=EvaluationVLLMBackend)``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from multiprocessing.connection import Connection
from typing import Any

from .util_for_types import ModelSpec
from .util_for_vllm import VLLMBackend, VLLMBackendError, _configure_environment, _render_chat_tokens


@dataclass(frozen=True)
class EvaluationRequest:
    request_id: str
    messages: list[dict[str, str]]
    system_prompt: str
    seed: int
    kind: str = "completion"
    answer: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    request_id: str
    text: str = ""
    score: float | None = None
    error: str | None = None
    error_code: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


def _render_continuations(
    tokenizer: Any, requests: Sequence[EvaluationRequest], mode: str
) -> list[list[int]]:
    """Render the full final assistant content without its end-of-turn marker."""
    for request in requests:
        if not request.messages or request.messages[-1].get("role") != "assistant":
            raise ValueError("Evaluation continuation must end with an assistant message")
    return _render_chat_tokens(
        tokenizer, requests, mode, add_generation_prompt=False, continue_final_message=True,
    )


def _reachability_tokens(
    tokenizer: Any, request: EvaluationRequest, mode: str
) -> tuple[list[int], int]:
    if not isinstance(request.answer, str) or not request.answer.strip():
        raise ValueError("Reachability requires a nonblank reference answer")
    messages = [dict(message) for message in request.messages]
    if not messages:
        raise ValueError("Reachability requires at least one conversation message")
    phrase = "So, the answer is"
    if messages[-1].get("role") == "user":
        messages.append({"role": "assistant", "content": f"{phrase} "})
    elif messages[-1].get("role") == "assistant":
        messages[-1]["content"] += f" {phrase} "
    else:
        raise ValueError("Conversation roles must be user or assistant")
    full_messages = [dict(message) for message in messages]
    full_messages[-1]["content"] += request.answer
    prefix, full = _render_continuations(
        tokenizer,
        [replace(request, messages=messages), replace(request, messages=full_messages)],
        mode,
    )
    # Tokenization may merge the boundary space with the first answer token.
    start = 0
    for left, right in zip(prefix, full):
        if left != right:
            break
        start += 1
    if start == 0 or start >= len(full):
        raise ValueError("Could not locate the reference answer token span")
    return full, start


def _score_output(
    request: EvaluationRequest, output: Any, tokens: list[int], start: int
) -> EvaluationResult:
    details = {
        "prompt_token_count": len(tokens),
        "answer_token_count": len(tokens) - start,
        "scored_token_count": 0,
        "skipped_token_count": len(tokens) - start,
    }
    if list(getattr(output, "prompt_token_ids", None) or []) != tokens:
        return EvaluationResult(
            request.request_id,
            error="Scoring output did not preserve the prompt token IDs",
            error_code="invalid_output",
            details=details,
        )
    logprobs = getattr(output, "prompt_logprobs", None)
    values = []
    if logprobs is not None:
        for index in range(start, len(tokens)):
            entry = logprobs[index] if index < len(logprobs) else None
            # Preserve the original partial-logprob policy, but expose coverage.
            if entry is None or tokens[index] not in entry:
                continue
            try:
                value = float(entry[tokens[index]].logprob)
            except (AttributeError, TypeError, ValueError):
                return EvaluationResult(
                    request.request_id,
                    error="A reference answer token has an invalid log probability",
                    error_code="invalid_output",
                    details=details,
                )
            if not math.isfinite(value):
                return EvaluationResult(
                    request.request_id,
                    error="A reference answer token has a nonfinite log probability",
                    error_code="nonfinite_score",
                    details=details,
                )
            values.append(value)
    details["scored_token_count"] = len(values)
    details["skipped_token_count"] = details["answer_token_count"] - len(values)
    if not values:
        return EvaluationResult(
            request.request_id,
            error="No reference answer token log probabilities were available",
            error_code="prompt_logprobs",
            details=details,
        )
    score = sum(values) / len(values)
    if not math.isfinite(score):
        return EvaluationResult(
            request.request_id,
            error="Mean reference answer log probability is nonfinite",
            error_code="nonfinite_score",
            details=details,
        )
    return EvaluationResult(request.request_id, score=score, details=details)


def _generate_evaluation(
    llm: Any, sampling_type: Any, spec: ModelSpec, requests: Sequence[EvaluationRequest]
) -> list[EvaluationResult]:
    tokenizer = llm.get_tokenizer()
    max_model_len = llm.llm_engine.model_config.max_model_len
    mode = spec.options.get("system_prompt_mode", "auto")
    results: list[EvaluationResult | None] = [None] * len(requests)
    safe_indices, prompts, sampling, answer_starts = [], [], [], []
    for index, request in enumerate(requests):
        try:
            if request.kind == "reachability":
                tokens, answer_start = _reachability_tokens(tokenizer, request, mode)
                options = {"temperature": 0.0, "max_tokens": 1, "prompt_logprobs": 1, "n": 1}
            elif request.kind == "completion":
                tokens = _render_continuations(tokenizer, [request], mode)[0]
                answer_start = None
                options = dict(spec.generation)
                options.setdefault("temperature", 0.0)
                options.setdefault("max_tokens", 1000)
                options["n"] = 1
            else:
                raise ValueError(f"Unknown evaluation request kind: {request.kind}")
            options["seed"] = request.seed
            max_tokens = options["max_tokens"]
            if len(tokens) + max_tokens > max_model_len:
                results[index] = EvaluationResult(
                    request.request_id,
                    error=f"Prompt ({len(tokens)} tokens) plus generation budget ({max_tokens}) exceeds context ({max_model_len})",
                    error_code="context_length",
                    details={"prompt_token_count": len(tokens), "max_tokens": max_tokens},
                )
                continue
            params = sampling_type(**options)
        except Exception as exc:
            results[index] = EvaluationResult(
                request.request_id, error=f"{type(exc).__name__}: {exc}", error_code="invalid_request"
            )
            continue
        prompts.append({"prompt_token_ids": tokens})
        sampling.append(params)
        safe_indices.append(index)
        answer_starts.append(answer_start)
    if prompts:
        outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
        if len(outputs) != len(safe_indices):
            raise VLLMBackendError("vLLM returned a different number of evaluation outputs than requests")
        for index, output, prompt, answer_start in zip(safe_indices, outputs, prompts, answer_starts):
            request = requests[index]
            if answer_start is not None:
                results[index] = _score_output(request, output, prompt["prompt_token_ids"], answer_start)
                continue
            if len(output.outputs) != 1:
                raise VLLMBackendError("vLLM must return exactly one completion per evaluation request")
            completion = output.outputs[0]
            results[index] = EvaluationResult(
                request.request_id,
                text=completion.text,
                details={
                    "finish_reason": completion.finish_reason,
                    "prompt_token_count": len(prompt["prompt_token_ids"]),
                },
            )
    if any(result is None for result in results):
        raise VLLMBackendError("Incomplete evaluation result mapping")
    return results  # type: ignore[return-value]


def _evaluation_worker(connection: Connection, spec: ModelSpec) -> None:
    llm = None
    try:
        if os.name == "posix":
            os.setsid()
        connection.send(("started", os.getpid()))
        _configure_environment(spec)
        connection.send(("startup", "importing vLLM"))
        # CUDA is imported only after entering the owned child process.
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
                raise ValueError("Unknown evaluation worker operation")
            connection.send(("result", _generate_evaluation(llm, SamplingParams, spec, payload)))
    except EOFError:
        pass
    except BaseException as exc:
        try:
            connection.send(("error", f"{type(exc).__name__}: {exc}"))
        except (EOFError, OSError):
            pass
    finally:
        if llm is not None:
            try:
                shutdown = getattr(llm, "shutdown", None)
                if callable(shutdown):
                    shutdown()
                else:
                    executor = getattr(getattr(llm, "llm_engine", None), "model_executor", None)
                    shutdown = getattr(executor, "shutdown", None)
                    if callable(shutdown):
                        shutdown()
            except Exception:
                pass
        connection.close()


class EvaluationVLLMBackend(VLLMBackend):
    """Use the existing process lifecycle and request-ID validation for evaluation."""

    def __init__(self, spec: ModelSpec):
        super().__init__(spec, worker_target=_evaluation_worker)

    def generate(self, requests: Sequence[EvaluationRequest]) -> list[EvaluationResult]:
        return super().generate(requests)  # type: ignore[arg-type, return-value]
