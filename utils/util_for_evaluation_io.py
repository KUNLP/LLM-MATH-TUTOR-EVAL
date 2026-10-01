"""Read completed simulator snapshots and persist isolated evaluation runs."""

from __future__ import annotations

import fcntl
import logging
import math
import re
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .util import fingerprint, read_json, write_json
from .util_for_evaluation_config import semantic_model_options
from .util_for_speaker import DIALOGUE_FINISH_SYMBOL, DIALOGUE_UNFINISH_SYMBOL
from .util_for_types import ModelSpec

POLICY_VERSION = 2
ANNOTATION_STAGES = ("overinformative", "answer_reachability_gain", "teacher_judgment")
ERROR_STAGES = {"overinformative": 0, "reachability": 1, "teacher_judgment": 2}
logger = logging.getLogger(__name__)
_RESUME_TIMEOUTS = {"startup_timeout", "request_timeout", "shutdown_timeout"}


def _options_without_timeouts(options: dict) -> dict:
    if not isinstance(options, dict):
        raise ValueError("Evaluation model options must be a mapping")
    # Deliberately do not recurse: engine/generation settings with a similarly
    # named field still affect identity and must not become resume overrides.
    return {key: deepcopy(value) for key, value in options.items() if key not in _RESUME_TIMEOUTS}


def _model_identity_without_timeouts(spec: dict) -> dict:
    if not isinstance(spec, dict):
        raise ValueError("Evaluation model identity must be a mapping")
    normalized = deepcopy(spec)
    if "options" in normalized:
        normalized["options"] = _options_without_timeouts(normalized["options"])
    return normalized


def _resume_identity(identity: dict) -> dict:
    """Remove only worker timeouts at the evaluator's known option locations."""
    normalized = deepcopy(identity)
    for field in ("vllm", "student_options"):
        if field in normalized:
            normalized[field] = _options_without_timeouts(normalized[field])
    if "checker" in normalized:
        normalized["checker"] = _model_identity_without_timeouts(normalized["checker"])
    if "students" in normalized:
        if not isinstance(normalized["students"], dict):
            raise ValueError("Evaluation student identities must be a mapping")
        normalized["students"] = {
            key: _model_identity_without_timeouts(spec)
            for key, spec in normalized["students"].items()
        }
    return normalized


@dataclass
class SourceRun:
    key: str
    path: Path
    teacher: str
    student: str
    target_data: str
    identity: dict
    problems: dict[int, dict]
    dialogues: dict[int, list[dict]]
    states: dict[int, dict]
    digest: str


def load_source(path: str | Path, config: dict, dataset_loader: Callable | None = None) -> SourceRun:
    """Prefer authoritative checkpoints; reject partial runs and dataset drift."""
    path = Path(path).resolve()
    lock_path = path / "_run.lock"
    lock = lock_path.open("r", encoding="utf-8") if lock_path.exists() else nullcontext()
    with lock as handle:
        if handle is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Simulator is still using {path}") from exc
        manifest = read_json(path / "manifest.json")
        if manifest.get("schema_version") != 1:
            raise ValueError(f"Unsupported simulator schema: {path}")
        identity = manifest["identity"]
        if manifest.get("fingerprint") != fingerprint(identity):
            raise ValueError(f"Invalid simulator manifest fingerprint: {path}")
        source_config = manifest["config"]
        args, paths = source_config["test_args"], source_config["paths"]
        if dataset_loader is None:
            from .util_for_math_data import get_math_datas

            dataset_loader = get_math_datas
        dataset = dataset_loader(
            name=args["target_data"],
            cache_dir=config["paths"]["cache_dir"],
            input_dir=paths["input_dir"],
            debug=args.get("debug", False),
            dataset_file=config["paths"].get("dataset_file") or paths.get("dataset_file"),
            limit=args.get("limit"),
            offline=config["evaluation"]["offline"],
        )
        if not dataset:
            raise ValueError(f"Source dataset is empty: {path}")
        problems = {}
        for problem in dataset:
            q_id = problem.get("id")
            if type(q_id) is not int or q_id < 0 or q_id in problems:
                raise ValueError("Dataset IDs must be unique non-negative integers")
            if any(not isinstance(problem.get(key), str) or not problem[key].strip()
                   for key in ("question", "answer")):
                raise ValueError(f"Missing question or answer for problem {q_id}")
            problems[q_id] = problem
        expected = fingerprint([
            {key: problem[key] for key in ("id", "question", "answer")} for problem in dataset
        ])
        if expected != identity.get("dataset_fingerprint"):
            raise ValueError(f"Dataset differs from the simulated problems: {path}")
        checkpoint_dir = path / "_state"
        if not checkpoint_dir.is_dir():
            raise ValueError(f"Expected a new simulator run with _state checkpoints: {path}")
        checkpoint_ids = {
            int(file.stem) for file in checkpoint_dir.iterdir()
            if file.is_file() and re.fullmatch(r"\d+\.json", file.name)
        }
        if checkpoint_ids != set(problems):
            raise ValueError(f"Checkpoint problem IDs do not match the source dataset: {path}")
        states = {q_id: read_json(checkpoint_dir / f"{q_id}.json") for q_id in sorted(problems)}
        dialogues = {}
        for q_id, state in states.items():
            if state.get("q_id") != q_id or state.get("phase") != "done":
                raise ValueError(f"Problem {q_id} has an invalid or incomplete checkpoint: {path}")
            messages = state.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"Invalid dialogue for problem {q_id}")
            if type(state.get("solved")) is not bool:
                raise ValueError(f"Missing solved status for problem {q_id}")
            marker = DIALOGUE_FINISH_SYMBOL if state["solved"] else DIALOGUE_UNFINISH_SYMBOL
            if messages[-1] != {"role": "system", "content": marker}:
                raise ValueError(f"Invalid final marker for problem {q_id}")
            for index, utterance in enumerate(messages[:-1]):
                if (utterance.get("role") != ("teacher" if index % 2 == 0 else "student")
                        or not isinstance(utterance.get("content"), str)):
                    raise ValueError(f"Invalid turn {index} for problem {q_id}")
                if utterance["role"] == "student" and type(utterance.get("is_correct")) is not bool:
                    raise ValueError(f"Missing simulator correctness for problem {q_id}, turn {index}")
            if state["solved"] != any(item.get("is_correct", False) for item in messages):
                raise ValueError(f"Correctness and solved status disagree for problem {q_id}")
            dialogues[q_id] = messages
        digest = fingerprint({"identity": identity, "states": states})
        teacher = identity["models"]["teacher"]["name"]
        student = identity["models"]["student"]["name"]
        return SourceRun(
            key=fingerprint(str(path))[:16], path=path, teacher=teacher, student=student,
            target_data=args["target_data"], identity=identity, problems=problems,
            dialogues=dialogues, states=states, digest=digest,
        )


def validate_comparison_sources(sources: list[SourceRun], *, comparison_mode: str = "model") -> None:
    """Do not normalize runs with different problems, budgets or student models."""
    pairs, groups, teachers = set(), {}, {}
    if comparison_mode not in {"model", "unrestricted"}:
        raise ValueError("comparison_mode must be model or unrestricted")
    for source in sources:
        pair = (source.target_data, source.teacher, source.student)
        if pair in pairs:
            raise ValueError(f"Select only one run per dataset/teacher/student combination: {pair}")
        pairs.add(pair)
        identity = source.identity
        student = identity["models"]["student"]
        comparison = {
            "dataset": identity["dataset_fingerprint"],
            "budget": identity["max_feedback_count"],
            "final_teacher": identity["collect_final_teacher_response"],
            "student": {key: student[key] for key in ("name", "model", "backend", "generation")},
            "student_options": semantic_model_options(student.get("options", {})),
            "student_prompt": identity["prompts"]["student"],
            "seed": identity["seed"],
            "verifier": identity.get("verifier"),
            "prompt_rendering_version": identity.get("prompt_rendering_version"),
        }
        if comparison_mode == "model":
            comparison["teacher_prompt"] = identity["prompts"]["teacher"]
        group = (source.target_data, source.student)
        if group in groups and groups[group] != comparison:
            raise ValueError(f"Cannot compare different datasets or simulation settings for {group}")
        groups[group] = comparison
        teacher = identity["models"]["teacher"]
        teacher_identity = {
            **{key: teacher[key] for key in ("name", "model", "backend", "generation")},
            "options": semantic_model_options(teacher.get("options", {})),
            "prompt": identity["prompts"]["teacher"],
        }
        teacher_group = (source.target_data, source.teacher)
        if teacher_group in teachers and teachers[teacher_group] != teacher_identity:
            raise ValueError(f"Teacher name refers to different model settings: {teacher_group}")
        teachers[teacher_group] = teacher_identity


class EvaluationStore:
    """Atomic stage snapshots and per-request cache; no writes to source runs."""

    def __init__(self, path: Path, identity: dict, config: dict, resume: bool):
        self.path = path
        self._lock = None
        self._saved_identity = deepcopy(identity)
        if resume:
            if not path.is_dir():
                raise FileNotFoundError(f"Cannot resume missing evaluation: {path}")
        else:
            path.mkdir(parents=True, exist_ok=False)
        try:
            self._lock = (path / "_run.lock").open("a", encoding="utf-8")
            try:
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another evaluator is using {path}") from exc
            manifest_path = path / "manifest.json"
            if resume:
                manifest = read_json(manifest_path)
                saved_identity = manifest.get("identity") if isinstance(manifest, dict) else None
                if (not isinstance(saved_identity, dict)
                        or manifest.get("fingerprint") != fingerprint(saved_identity)):
                    raise ValueError("Invalid evaluation manifest fingerprint")
                if fingerprint(_resume_identity(saved_identity)) != fingerprint(_resume_identity(identity)):
                    raise ValueError("Evaluation resume refused: inputs, models or evaluation settings changed")
                # Keep the original manifest and cache namespaces intact. Only
                # the worker spec receives the new timeouts during this run.
                self._saved_identity = deepcopy(saved_identity)
            else:
                write_json({
                    "schema_version": 1, "policy_version": POLICY_VERSION,
                    "fingerprint": fingerprint(identity), "identity": identity,
                    "config": config, "created_at": datetime.now(UTC).isoformat(),
                }, manifest_path)
        except BaseException:
            self.close()
            raise

    @property
    def saved_identity(self) -> dict:
        """Return the immutable run's original model/cache identities."""
        return deepcopy(self._saved_identity)

    def stage(self, source: SourceRun, name: str) -> dict[int, list[dict]] | None:
        path = self.path / source.key / f"{name}.json"
        if not path.exists():
            return None
        payload = read_json(path)
        if payload.get("source_fingerprint") != source.digest:
            raise ValueError(f"Stale evaluation snapshot: {path}")
        dialogues = {int(key): value for key, value in payload["dialogues"].items()}
        if set(dialogues) != set(source.dialogues):
            raise ValueError(f"Evaluation snapshot problem IDs mismatch: {path}")
        return dialogues

    def save_stage(self, source: SourceRun, name: str, dialogues: dict[int, list[dict]]) -> None:
        write_json({"source_fingerprint": source.digest, "dialogues": dialogues},
                   self.path / source.key / f"{name}.json")

    def annotation_error_count(self, sources: list[SourceRun]) -> int:
        """Count each source's most complete snapshot, including earlier-stage errors."""
        count = 0
        for source in sources:
            for name in reversed(ANNOTATION_STAGES):
                annotated = self.stage(source, name)
                if annotated is not None:
                    count += sum(len(turn.get("annotation_errors", []))
                                 for dialogue in annotated.values() for turn in dialogue)
                    break
        return count

    def prepare_retry(self, sources: list[SourceRun], *, requested: bool) -> bool:
        """Journal exact failed cache keys before invalidating dependent artifacts.

        A pending journal is continued even without another retry flag. Once its
        invalidation phase finishes, resumed work retains newly repaired caches.
        """
        journal = self.path / "_retry.json"
        if journal.exists():
            plan = read_json(journal)
        elif requested:
            earliest, failures = {}, {}
            namespaces = {
                source.key: {
                    0: fingerprint(self.saved_identity["students"][source.key])
                       if source.key in self.saved_identity.get("students", {}) else None,
                    1: fingerprint(self.saved_identity["students"][source.key])
                       if source.key in self.saved_identity.get("students", {}) else None,
                    2: fingerprint(self.saved_identity["checker"]),
                } for source in sources
            }
            for source in sources:
                for name in ANNOTATION_STAGES:
                    annotated = self.stage(source, name)
                    if annotated is None:
                        continue
                    for dialogue in annotated.values():
                        for turn in dialogue:
                            for error in turn.get("annotation_errors", []):
                                stage = ERROR_STAGES[error["stage"]]
                                request_key = error.get("request_fingerprint", "")
                                model_key = namespaces[source.key][stage]
                                if not re.fullmatch(r"[0-9a-f]{64}", request_key) or model_key is None:
                                    raise ValueError("Cannot safely retry annotation without its exact request fingerprint")
                                earliest[source.key] = min(stage, earliest.get(source.key, stage))
                                failures[(model_key, request_key)] = stage
            # Markers also retain failures from a stage interrupted before its
            # annotation snapshot was saved. They carry exact cache identities.
            interrupted_failures = []
            for marker in (self.path / "_responses" / "_failed").glob("*/*.json"):
                interrupted_failures.append((marker.parent.name, marker.stem, read_json(marker)["request_id"]))
            # A process may die after a response is cached but before its
            # annotator publishes the failure marker. Recover detectable cases.
            from .util_for_evaluation_checkers import parse_label

            for response_path in (self.path / "_responses").glob("[0-9a-f]*/*.json"):
                response = read_json(response_path)["result"]
                request_id = response["request_id"]
                stage_name = request_id.split(":", 1)[0]
                if stage_name not in ERROR_STAGES:
                    continue
                failed = response.get("error") is not None or response.get("error_code") is not None
                if stage_name == "reachability":
                    score = response.get("score")
                    failed = failed or type(score) not in (int, float) or not math.isfinite(score)
                    failed = failed or response.get("details", {}).get("skipped_token_count", 0) > 0
                elif stage_name == "teacher_judgment":
                    failed = failed or parse_label(response.get("text")) is None
                else:
                    text = response.get("text")
                    failed = failed or not isinstance(text, str) or not text.strip()
                if failed:
                    interrupted_failures.append((response_path.parent.name, response_path.stem, request_id))
            for model_key, request_key, request_id in interrupted_failures:
                if not all(re.fullmatch(r"[0-9a-f]{64}", key) for key in (model_key, request_key)):
                    raise ValueError("Invalid failed-response cache identity")
                stage = ERROR_STAGES[request_id.split(":", 1)[0]]
                if (model_key, request_key) not in failures:
                    for source in sources:
                        if namespaces[source.key][stage] == model_key:
                            earliest[source.key] = min(stage, earliest.get(source.key, stage))
                failures[(model_key, request_key)] = stage
            if not failures:
                return False
            plan = {"phase": "invalidate", "sources": earliest,
                    "responses": [[model, request] for model, request in failures]}
            write_json(plan, journal)
        else:
            return False
        if (not isinstance(plan, dict) or not isinstance(plan.get("sources"), dict)
                or set(plan["sources"]) - {source.key for source in sources}
                or any(type(stage) is not int or not 0 <= stage < len(ANNOTATION_STAGES)
                       for stage in plan["sources"].values())
                or not isinstance(plan.get("responses"), list)
                or any(not isinstance(pair, list) or len(pair) != 2
                       or any(not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key) for key in pair)
                       for pair in plan["responses"])):
            raise ValueError("Invalid evaluation retry journal")
        if plan["phase"] == "invalidate":
            for model_key, request_key in plan["responses"]:
                (self.path / "_responses" / model_key / f"{request_key}.json").unlink(missing_ok=True)
                (self.path / "_responses" / "_failed" / model_key / f"{request_key}.json").unlink(missing_ok=True)
            for source_key, first_stage in plan["sources"].items():
                for name in ANNOTATION_STAGES[first_stage:]:
                    (self.path / source_key / f"{name}.json").unlink(missing_ok=True)
            for name in ("metrics.json", "metrics.csv", "metrics.xlsx"):
                (self.path / name).unlink(missing_ok=True)
            plan["phase"] = "regenerate"
            write_json(plan, journal)
        elif plan["phase"] != "regenerate":
            raise ValueError("Invalid evaluation retry journal")
        return True

    def finish_retry(self) -> None:
        (self.path / "_retry.json").unlink(missing_ok=True)

    def close(self) -> None:
        if self._lock is not None:
            self._lock.close()
            self._lock = None


class CachedEvaluationBackend:
    """Use original cache keys while allowing new worker timeouts on resume."""

    def __init__(
        self, spec: ModelSpec, path: Path, factory: Callable,
        *, cache_spec: ModelSpec | dict | None = None,
    ):
        self.spec = spec
        runtime_identity = asdict(spec)
        cache_identity = (
            runtime_identity if cache_spec is None
            else deepcopy(cache_spec) if isinstance(cache_spec, dict)
            else asdict(cache_spec)
        )
        if (fingerprint(_model_identity_without_timeouts(runtime_identity))
                != fingerprint(_model_identity_without_timeouts(cache_identity))):
            raise ValueError("Evaluation cache model differs beyond worker timeouts")
        self.path = path / fingerprint(cache_identity)
        self.failure_path = path / "_failed" / fingerprint(cache_identity)
        self.factory = factory
        self.backend = None

    def generate(self, requests):
        from .util_for_evaluation_backend import EvaluationResult

        if len({request.request_id for request in requests}) != len(requests):
            raise ValueError("Duplicate evaluation request IDs")
        results, pending, destinations = {}, [], {}
        for request in requests:
            identity = asdict(request)
            key = fingerprint(identity)
            destination = self.path / f"{key}.json"
            destinations[request.request_id] = (destination, key)
            if destination.exists():
                cached = read_json(destination)
                result = EvaluationResult(**cached["result"])
                if cached.get("fingerprint") != key or result.request_id != request.request_id:
                    raise ValueError(f"Invalid cached evaluation response: {destination}")
                results[request.request_id] = result
            else:
                pending.append(request)
        if pending:
            if self.backend is None:
                self.backend = self.factory(self.spec)
            generated = self.backend.generate(pending)
            expected = {request.request_id for request in pending}
            if len(generated) != len(expected) or {item.request_id for item in generated} != expected:
                raise RuntimeError("Evaluation backend returned duplicate, missing or unexpected IDs")
            for result in generated:
                destination, key = destinations[result.request_id]
                if result.error is not None or result.error_code is not None:
                    # Commit provenance first: a marker-write failure must not
                    # leave an unmarked failed result that ordinary resume reuses.
                    write_json({"request_id": result.request_id}, self.failure_path / f"{key}.json")
                write_json({"fingerprint": key, "result": asdict(result)}, destination)
                if result.error is None and result.error_code is None:
                    (self.failure_path / f"{key}.json").unlink(missing_ok=True)
                results[result.request_id] = result
        return [results[request.request_id] for request in requests]

    def mark_failed(self, request_fingerprint: str) -> None:
        """Record annotator-detected failures without discarding successful siblings."""
        response = read_json(self.path / f"{request_fingerprint}.json")
        write_json({"request_id": response["result"]["request_id"]},
                   self.failure_path / f"{request_fingerprint}.json")

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()
            self.backend = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except BaseException:
            if exc_type is None:
                raise
            logger.exception("Evaluation backend cleanup failed while handling another error")
