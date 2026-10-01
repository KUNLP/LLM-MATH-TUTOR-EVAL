"""Dependency-free contracts shared by the simulator and inference backends."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend: str
    model: str
    generation: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    messages: list[dict[str, str]]
    system_prompt: str
    seed: int


@dataclass(frozen=True)
class GenerationResult:
    request_id: str
    text: str = ""
    finish_reason: str | None = None
    error: str | None = None
    error_code: str | None = None


class Backend(Protocol):
    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]: ...

    def close(self) -> None: ...
