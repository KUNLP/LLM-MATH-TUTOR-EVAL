"""Resident, single-GPU replicas for data-parallel local inference."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
from threading import Lock

from .util_for_types import Backend, GenerationRequest, GenerationResult, ModelSpec


class DataParallelBackend:
    """Keep one complete model on each GPU and split independent requests.

    Each child owns its CUDA process and is called by at most one thread at a
    time. A dispatch waits for all active children before returning, including
    on failure, so shutdown never races a child's synchronous pipe operations.
    Failing dispatches signal cooperative cancellation before waiting; children
    without that capability are bounded by their request timeouts instead.
    """

    def __init__(
        self, spec: ModelSpec, factory: Callable[[ModelSpec], Backend] | None = None
    ):
        if spec.backend != "vllm":
            raise ValueError("Data parallelism requires a vllm model")
        mask = spec.options.get("cuda_visible_devices")
        if not isinstance(mask, str) or not mask.strip():
            raise ValueError("Data parallelism requires explicit cuda_visible_devices")
        devices = [device.strip() for device in mask.split(",")]
        if not all(devices) or len(set(devices)) != len(devices):
            raise ValueError("Data parallelism requires distinct, nonempty GPU devices")
        size = spec.options.get("data_parallel_size", len(devices))
        if type(size) is not int or size != len(devices):
            raise ValueError("data_parallel_size must match the number of assigned GPUs")
        engine = spec.options.get("engine_kwargs", {})
        if not isinstance(engine, dict):
            raise ValueError("engine_kwargs must be a mapping")
        for key in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"):
            value = engine.get(key, 1)
            if type(value) is not int or value != 1:
                raise ValueError(f"Each data-parallel replica requires engine_kwargs.{key}=1")

        if factory is None:
            from .util_for_vllm import VLLMBackend

            factory = VLLMBackend
        self.spec = spec
        self._replicas: list[Backend | None] = []
        self._lock = Lock()
        self._closed = False
        self._failed = False
        try:
            for device in devices:
                options = deepcopy(spec.options)
                options["cuda_visible_devices"] = device
                options["data_parallel_size"] = 1
                options.setdefault("engine_kwargs", {})["tensor_parallel_size"] = 1
                replica_spec = replace(spec, generation=deepcopy(spec.generation), options=options)
                self._replicas.append(factory(replica_spec))
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                error.add_note(f"Failed to clean up initialized DP replicas: {cleanup_error}")
            raise

    @staticmethod
    def _generate_shard(
        backend: Backend, requests: Sequence[GenerationRequest]
    ) -> list[GenerationResult]:
        results = backend.generate(requests)
        expected = {request.request_id for request in requests}
        received = [result.request_id for result in results]
        if len(received) != len(expected) or set(received) != expected:
            raise RuntimeError("DP replica returned duplicate, missing or unexpected request IDs")
        return results

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        if not requests:
            return []
        expected = [request.request_id for request in requests]
        if len(set(expected)) != len(expected):
            raise ValueError("Duplicate inference request IDs")
        with self._lock:
            if self._closed:
                raise RuntimeError("Data-parallel backend is closed")
            if self._failed:
                raise RuntimeError("Data-parallel backend failed; start a new run to resume")
            replicas = self._replicas
            shards = [list(requests[index::len(replicas)]) for index in range(len(replicas))]
            results: dict[str, GenerationResult] = {}
            # Keeping the executor inside the lock guarantees that close() or
            # another dispatch cannot touch a replica while its thread runs.
            with ThreadPoolExecutor(
                max_workers=min(len(replicas), len(requests)), thread_name_prefix="vllm-dp"
            ) as executor:
                futures = []
                try:
                    for replica, shard in zip(replicas, shards):
                        if shard:
                            assert replica is not None
                            futures.append(executor.submit(self._generate_shard, replica, shard))
                    for future in as_completed(futures):
                        results.update((result.request_id, result) for result in future.result())
                except BaseException as error:
                    self._failed = True
                    for future in futures:
                        future.cancel()
                    # Only signal cancellation here. Each child's generating
                    # thread owns its pipe and performs its own shutdown after
                    # observing the signal, avoiding concurrent close/read.
                    for index, replica in enumerate(replicas):
                        try:
                            cancel = getattr(replica, "cancel_pending", None)
                            if callable(cancel):
                                cancel()
                        except BaseException as cancellation_error:
                            error.add_note(
                                f"DP replica {index} cancellation failed: {cancellation_error}"
                            )
                    # __exit__ joins all running calls before releasing the lock.
                    raise
            return [results[request_id] for request_id in expected]

    def close(self) -> None:
        with self._lock:
            self._closed = True
            first_error: BaseException | None = None
            for index, replica in enumerate(self._replicas):
                if replica is None:
                    continue
                try:
                    replica.close()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
                    else:
                        first_error.add_note(f"Another DP replica failed to close: {error}")
                else:
                    # Retain only failed closes, allowing a later close retry
                    # without touching workers that were successfully reaped.
                    self._replicas[index] = None
            if first_error is not None:
                raise first_error
