"""Optional, validated sharing of first student responses across teacher runs."""

from __future__ import annotations

import fcntl
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from .util import fingerprint, read_json, write_json
from .util_for_types import GenerationRequest, GenerationResult, ModelSpec

# Execution placement and transport do not change the requested student policy.
_OPERATIONAL_OPTIONS = {
    "cache_dir", "hf_token_file", "api_key_file", "api_key_env", "offline",
    "cuda_visible_devices", "data_parallel_size", "startup_timeout", "request_timeout",
    "shutdown_timeout", "mode", "max_concurrency", "timeout", "max_retries",
    "poll_interval", "batch_timeout",
}


def initial_response_identity(
    spec: ModelSpec, *, dataset_name: str, instruction: str, seed: int,
    verifier: dict, prompt_rendering_version: int,
) -> dict:
    """Exclude teachers and subsets; each entry validates its complete problem."""
    model = asdict(spec)
    model["options"] = {
        key: value for key, value in model["options"].items()
        if key not in _OPERATIONAL_OPTIONS
    }
    return {
        "schema_version": 1,
        "student": model,
        "dataset_name": dataset_name,
        "student_instruction": instruction,
        "seed": seed,
        "verifier": verifier,
        "prompt_rendering_version": prompt_rendering_version,
    }


class InitialResponseCache:
    """Atomic per-problem records protected through generation and grading."""

    def __init__(self, root: Path, identity: dict):
        self.identity_fingerprint = fingerprint(identity)
        self.path = Path(root) / self.identity_fingerprint

    def _key(self, problem: dict, request: GenerationRequest) -> str:
        return fingerprint({
            "problem": {key: problem[key] for key in ("id", "question", "answer")},
            "request": asdict(request),
        })

    @contextmanager
    def locked(self, entries: list[tuple[dict, GenerationRequest]]) -> Iterator[None]:
        if not entries:
            yield
            return
        self.path.mkdir(parents=True, exist_ok=True)
        handles = []
        try:
            # A common order prevents overlapping batches from deadlocking.
            for key in sorted({self._key(problem, request) for problem, request in entries}):
                handle = (self.path / f"{key}.lock").open("a", encoding="utf-8")
                handles.append(handle)
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            for handle in reversed(handles):
                handle.close()

    def load(self, problem: dict, request: GenerationRequest) -> GenerationResult | None:
        key = self._key(problem, request)
        path = self.path / f"{key}.json"
        if not path.exists():
            return None
        saved = read_json(path)
        if (
            saved.get("identity_fingerprint") != self.identity_fingerprint
            or saved.get("entry_fingerprint") != key
            or type(saved.get("is_correct")) is not bool
        ):
            raise ValueError("Invalid shared initial-response cache identity")
        result = GenerationResult(**saved["result"])
        if result.request_id != request.request_id or result.error or not result.text.strip():
            raise ValueError("Invalid shared initial-response cache result")
        return result

    def save(
        self, problem: dict, request: GenerationRequest,
        result: GenerationResult, is_correct: bool, *, overwrite: bool = False,
    ) -> None:
        if result.error or not result.text.strip() or type(is_correct) is not bool:
            raise ValueError("Only successfully generated and graded responses may be cached")
        if not overwrite and self.load(problem, request) is not None:
            return
        key = self._key(problem, request)
        write_json({
            "identity_fingerprint": self.identity_fingerprint,
            "entry_fingerprint": key,
            "is_correct": is_correct,
            "result": asdict(result),
        }, self.path / f"{key}.json")
