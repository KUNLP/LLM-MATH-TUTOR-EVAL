"""Keep each role's API backend or GPU replica pool alive for the entire run."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from .util_for_config import validate_model_allocations
from .util_for_types import Backend, GenerationRequest, GenerationResult, ModelSpec

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(
        self,
        specs: dict[str, ModelSpec],
        work_dir: Path,
        factories: dict[str, Callable] | None = None,
    ):
        validate_model_allocations(specs)
        self.specs = specs
        self.work_dir = Path(work_dir)
        self.factories = factories or {}
        self._backends: dict[str, Backend] = {}
        self._closed = False

    def _build(self, spec: ModelSpec, role: str) -> Backend:
        if spec.backend == "vllm":
            from .util_for_dp import DataParallelBackend

            return DataParallelBackend(spec, factory=self.factories.get("vllm"))
        if spec.backend in self.factories:
            return self.factories[spec.backend](spec)
        from .util_for_api import APIBackend

        return APIBackend(spec, self.work_dir / "_api" / role)

    def _backend(self, role: str) -> Backend:
        spec = self.specs[role]
        if role not in self._backends:
            if spec.backend == "vllm":
                logger.info(
                    "Loading %s model %s on GPUs %s; replicas stay resident until shutdown",
                    role, spec.name, spec.options["cuda_visible_devices"],
                )
            self._backends[role] = self._build(spec, role)
        return self._backends[role]

    def generate(self, role: str, requests: list[GenerationRequest]) -> list[GenerationResult]:
        if self._closed:
            raise RuntimeError("Model manager is closed")
        if not requests:
            return []
        expected = [request.request_id for request in requests]
        if len(set(expected)) != len(expected):
            raise ValueError("Duplicate inference request IDs")
        results = self._backend(role).generate(requests)
        by_id = {result.request_id: result for result in results}
        if len(results) != len(by_id) or set(by_id) != set(expected):
            raise RuntimeError("Backend returned duplicate, missing or unexpected request IDs")
        return [by_id[request_id] for request_id in expected]

    def close(self) -> None:
        self._closed = True
        error: BaseException | None = None
        for role, backend in list(self._backends.items()):
            try:
                backend.close()
            except BaseException as exc:
                error = error or exc
                logger.exception("Failed to close the %s backend", role)
            else:
                # Keep only failed backends so a later close can retry cleanup.
                del self._backends[role]
        if error:
            raise error

    def __enter__(self) -> ModelManager:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.close()
        except BaseException:
            if exc_type is None:
                raise
            logger.exception("Backend cleanup failed while handling another error")
