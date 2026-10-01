"""Batched teacher/student tutoring with persistent per-role GPU replicas.

No model, credentials, dataset or CUDA runtime is loaded during import.
"""

from __future__ import annotations

import fcntl
import logging
import signal
import uuid
from collections import Counter
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from utils.util import fingerprint, read_json, write_json
from utils.util_for_config import (
    normalize_config,
    prepare_hydra_config,
    register_hydra_resolvers,
    resolve_model_spec,
    validate_component,
)
from utils.util_for_instruction import get_start_utterance
from utils.util_for_initial_state import InitialResponseCache, initial_response_identity
from utils.util_for_policies import PROMPT_RENDERING_VERSION, get_verifier_identity
from utils.util_for_simulator import ModelManager
from utils.util_for_speaker import DIALOGUE_FINISH_SYMBOL, DIALOGUE_UNFINISH_SYMBOL, Speaker
from utils.util_for_types import GenerationRequest, GenerationResult

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1
# Register on import as well: Hydra must resolve paths before it calls main.
register_hydra_resolvers()


@dataclass
class DialogueState:
    q_id: int
    messages: list[dict]
    phase: str = "student"
    feedback_count: int = 0
    solved: bool = False
    stop_reason: str | None = None
    initial_response_fingerprint: str | None = None
    errors: list[dict] = field(default_factory=list)

    def finish(self, reason: str) -> None:
        self.phase = "done"
        self.stop_reason = reason
        self.messages.append(
            {
                "role": "system",
                "content": DIALOGUE_FINISH_SYMBOL if self.solved else DIALOGUE_UNFINISH_SYMBOL,
            }
        )


class RunStore:
    """A run has its own directory; existing runs are only opened with resume."""

    def __init__(self, path: Path, identity: dict, config: dict, resume: bool):
        self.path = path
        self._lock = None
        if resume:
            if not path.is_dir():
                raise FileNotFoundError(f"Cannot resume a missing run: {path}")
        else:
            path.mkdir(parents=True, exist_ok=False)
        try:
            self._lock = (path / "_run.lock").open("a", encoding="utf-8")
            try:
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another simulator is using this run: {path}") from exc
            manifest_path = path / "manifest.json"
            if resume:
                manifest = read_json(manifest_path)
                if manifest.get("fingerprint") != fingerprint(identity):
                    raise ValueError(
                        "Resume refused: dataset, model, prompt or simulation settings changed"
                    )
            else:
                write_json(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "fingerprint": fingerprint(identity),
                        "identity": identity,
                        "config": config,
                        "created_at": datetime.now(UTC).isoformat(),
                    },
                    manifest_path,
                )
            (path / "_state").mkdir(exist_ok=True)
        except BaseException:
            self.close()
            raise

    def save(self, state: DialogueState) -> None:
        # The checkpoint is authoritative; exports can be regenerated after a crash.
        write_json(asdict(state), self.path / "_state" / f"{state.q_id}.json")
        write_json(state.messages, self.path / f"{state.q_id}.json")

    def load(self, q_id: int) -> DialogueState | None:
        path = self.path / "_state" / f"{q_id}.json"
        if not path.exists():
            return None
        saved = read_json(path)
        # Older checkpoints may contain retired metadata; it has no runtime effect.
        saved.pop("initial_cache_hit", None)
        state = DialogueState(**saved)
        if state.q_id != q_id or state.phase not in {"student", "teacher", "final_teacher", "done"}:
            raise ValueError(f"Invalid checkpoint for problem {q_id}")
        dialogue = state.messages[:-1] if state.phase == "done" else state.messages
        if not dialogue or any(
            utterance.get("role") != ("teacher" if i % 2 == 0 else "student")
            or not isinstance(utterance.get("content"), str)
            for i, utterance in enumerate(dialogue)
        ):
            raise ValueError(f"Invalid dialogue in checkpoint for problem {q_id}")
        if state.phase == "done":
            expected = DIALOGUE_FINISH_SYMBOL if state.solved else DIALOGUE_UNFINISH_SYMBOL
            if state.messages[-1] != {"role": "system", "content": expected}:
                raise ValueError(f"Invalid end marker for problem {q_id}")
        elif (len(dialogue) % 2 == 1) != (state.phase == "student"):
            raise ValueError(f"Invalid next role in checkpoint for problem {q_id}")
        return state

    def close(self) -> None:
        if self._lock is not None:
            self._lock.close()
            self._lock = None


class TutoringSimulator:
    def __init__(
        self,
        config: dict,
        *,
        dataset: list[dict] | None = None,
        manager: ModelManager | None = None,
        verifier: Callable | None = None,
        run_id: str | None = None,
        resume: bool = False,
    ):
        root = config.get("paths", {}).get("root_dir") or Path(__file__).resolve().parent
        self.config = normalize_config(config, root)
        self.args = self.config["test_args"]
        self.specs = {
            role: resolve_model_spec(self.config, role) for role in ("teacher", "student")
        }
        self.speakers = {
            role: Speaker(role, self.config["prompts"][role], self.args["seed"])
            for role in self.specs
        }
        self.dataset = dataset
        self.manager = manager
        self.verifier = verifier
        if resume and not run_id:
            raise ValueError("run.resume=true requires an explicit run.id")
        self.run_id = validate_component(
            run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8],
            "run_id",
        )
        self.resume = resume
        self.output_dir = (
            Path(self.config["paths"]["output_dir"])
            / "from_tutoring_simulator"
            / self.specs["teacher"].name
            / self.specs["student"].name
            / self.args["target_data"]
            / self.run_id
        )
        self.states: dict[int, DialogueState] = {}
        self.problems: dict[int, dict] = {}
        self.store: RunStore | None = None
        self.initial_cache: InitialResponseCache | None = None

    def _load_dataset(self) -> None:
        if self.dataset is None:
            from utils.util_for_math_data import get_math_datas

            paths = self.config["paths"]
            self.dataset = get_math_datas(
                name=self.args["target_data"],
                cache_dir=paths["cache_dir"],
                input_dir=paths["input_dir"],
                debug=self.args["debug"],
                dataset_file=paths.get("dataset_file"),
                limit=self.args.get("limit"),
                offline=self.args["offline"],
            )
        if not self.dataset:
            raise ValueError("The dataset is empty")
        for problem in self.dataset:
            q_id = problem.get("id")
            if type(q_id) is not int or q_id < 0 or q_id in self.problems:
                raise ValueError("Problem IDs must be unique non-negative integers")
            for key in ("question", "answer"):
                if not isinstance(problem.get(key), str) or not problem[key].strip():
                    raise ValueError(f"Problem {q_id} needs a non-empty {key}")
            self.problems[q_id] = problem

    def _accept(
        self, state: DialogueState, role: str, result: GenerationResult, *,
        initial_request: GenerationRequest | None = None,
    ) -> None:
        if result.error or not result.text.strip():
            code = result.error_code or "empty_response"
            state.errors.append(
                {"role": role, "code": code, "message": result.error or "Empty response"}
            )
            # A failed closing teacher response must never undo a correct student answer.
            state.finish("solved" if state.solved else code)
        elif role == "student":
            # Grade now, independently of the next teacher call or turn limit.
            correct = self.verifier(result.text, self.problems[state.q_id]["answer"])
            if type(correct) is not bool:
                raise TypeError("The answer verifier must return a boolean")
            if self.initial_cache and initial_request is not None:
                # Publish only after successful grading, before advancing the checkpoint.
                # The run journal already preserves this exact response if publication fails.
                self.initial_cache.save(
                    self.problems[state.q_id], initial_request, result, correct,
                    overwrite=self.args["initial_response_mode"] == "refresh",
                )
            if len(state.messages) == 1:
                state.initial_response_fingerprint = fingerprint({
                    "problem": {key: self.problems[state.q_id][key] for key in ("id", "question", "answer")},
                    "content": result.text, "is_correct": correct,
                })
            state.messages.append(
                {"role": "student", "content": result.text, "is_correct": correct}
            )
            state.solved = correct
            at_limit = state.feedback_count >= self.args["max_feedback_count"]
            if correct or at_limit:
                if self.args["collect_final_teacher_response"]:
                    state.phase = "final_teacher"
                else:
                    state.finish("solved" if correct else "max_feedback_count")
            else:
                state.phase = "teacher"
        else:
            utterance = {"role": "teacher", "content": result.text}
            if state.phase == "final_teacher":
                utterance["is_final_response"] = True
                state.messages.append(utterance)
                state.finish("solved" if state.solved else "max_feedback_count")
            else:
                state.messages.append(utterance)
                state.feedback_count += 1
                state.phase = "student"
        self.store.save(state)

    def _complete_dispatch(self, path: Path, dispatch: dict) -> list[GenerationResult]:
        role = dispatch["role"]
        requests = [GenerationRequest(**item) for item in dispatch["requests"]]
        expected = {request.request_id for request in requests}
        if len(expected) != len(requests) or not expected:
            raise ValueError("Invalid saved inference dispatch")
        if dispatch.get("results") is None:
            generated = self.manager.generate(role, requests)
            if {result.request_id for result in generated} != expected or len(generated) != len(
                expected
            ):
                raise RuntimeError("Inference results did not match the requested problems")
            dispatch["results"] = [asdict(result) for result in generated]
            # Commit the whole result group before writing any individual result.
            write_json(dispatch, path)
        results = [GenerationResult(**item) for item in dispatch["results"]]
        if {result.request_id for result in results} != expected or len(results) != len(expected):
            raise ValueError("Invalid saved inference results")
        request_lookup = {request.request_id: request for request in requests}
        for result in results:
            write_json(
                {
                    "request_fingerprint": fingerprint(asdict(request_lookup[result.request_id])),
                    "result": asdict(result),
                },
                self.output_dir / "_responses" / f"{result.request_id}.json",
            )
        # Once every individual response is durable, discard duplicate histories.
        write_json({"role": role, "request_ids": sorted(expected), "journaled": True}, path)
        return results

    def _dispatch(self, role: str, requests: list[GenerationRequest]) -> list[GenerationResult]:
        if not requests:
            return []
        dispatch = {"role": role, "requests": [asdict(request) for request in requests]}
        path = self.output_dir / "_dispatches" / f"{fingerprint(dispatch)}.json"
        if path.exists():
            raise RuntimeError(
                "An existing inference dispatch must be recovered before new requests"
            )
        # The exact request group is durable before submission, so resuming a
        # pending Batch never changes its hash when batch_size changes.
        write_json(dispatch, path)
        return self._complete_dispatch(path, dispatch)

    def _recover_dispatches(self) -> None:
        for path in sorted((self.output_dir / "_dispatches").glob("*.json")):
            dispatch = read_json(path)
            if not dispatch.get("journaled", False):
                logger.info("Recovering %s inference group", dispatch["role"])
                initial = {}
                if self.initial_cache and dispatch["role"] == "student":
                    pending = {
                        self.speakers["student"].request(state.q_id, state.messages).request_id: state
                        for state in self.states.values()
                        if state.phase == "student" and len(state.messages) == 1
                    }
                    for item in dispatch["requests"]:
                        request = GenerationRequest(**item)
                        state = pending.get(request.request_id)
                        if state is not None:
                            expected = self.speakers["student"].request(state.q_id, state.messages)
                            if request != expected:
                                raise ValueError("Pending initial dispatch does not match its checkpoint")
                            initial[request.request_id] = (state, request)
                entries = [
                    (self.problems[state.q_id], request) for state, request in initial.values()
                ]
                guard = self.initial_cache.locked(entries) if self.initial_cache else nullcontext()
                with guard:
                    # Recover the exact submitted group, including an in-flight API Batch.
                    # A shared cache populated during downtime cannot replace that work.
                    results = self._complete_dispatch(path, dispatch)
                    for result in results:
                        if result.request_id in initial:
                            state, request = initial[result.request_id]
                            self._accept(state, "student", result, initial_request=request)

    def _run_role(self, role: str) -> None:
        phases = {"student"} if role == "student" else {"teacher", "final_teacher"}
        pending = [state for state in self.states.values() if state.phase in phases]
        if not pending:
            return
        logger.info("%s: %d conversations (%s)", role, len(pending), self.specs[role].backend)
        size = self.args["batch_size"]
        # Derive the label from checkpoints so resumed runs keep their turn numbers.
        turns = {state.feedback_count + 1 for state in pending}
        first_turn, last_turn = min(turns), max(turns)
        turn_label = str(first_turn) if first_turn == last_turn else f"{first_turn}-{last_turn}"
        with tqdm(
            total=len(pending),
            desc=f"{role} turn {turn_label}",
            unit="dialogue",
            dynamic_ncols=True,
            leave=True,
        ) as progress:
            for offset in range(0, len(pending), size):
                states = pending[offset : offset + size]
                requests = [
                    self.speakers[role].request(state.q_id, state.messages) for state in states
                ]
                initial = {
                    request.request_id: self.problems[state.q_id]
                    for state, request in zip(states, requests)
                    if role == "student" and len(state.messages) == 1
                }
                entries = [(initial[r.request_id], r) for r in requests if r.request_id in initial]
                guard = self.initial_cache.locked(entries) if self.initial_cache else nullcontext()
                with guard:
                    results = {}
                    missing = []
                    for request in requests:
                        path = self.output_dir / "_responses" / f"{request.request_id}.json"
                        if path.exists():
                            # A run's durable response always wins, including refresh/resume.
                            saved = read_json(path)
                            if saved.get("request_fingerprint") != fingerprint(asdict(request)):
                                raise ValueError(f"Response journal mismatch: {request.request_id}")
                            result = GenerationResult(**saved["result"])
                            if result.request_id != request.request_id:
                                raise ValueError("Response journal ID mismatch")
                            results[request.request_id] = result
                            continue
                        cached = None
                        if (
                            self.initial_cache and request.request_id in initial
                            and self.args["initial_response_mode"] == "reuse"
                        ):
                            cached = self.initial_cache.load(initial[request.request_id], request)
                        if cached is None:
                            missing.append(request)
                        else:
                            # Capture the chosen shared response before grading/checkpointing.
                            write_json({
                                "request_fingerprint": fingerprint(asdict(request)),
                                "result": asdict(cached),
                            }, path)
                            results[request.request_id] = cached
                    for result in self._dispatch(role, missing):
                        results[result.request_id] = result
                    for state, request in zip(states, requests):
                        result = results[request.request_id]
                        self._accept(
                            state, role, result,
                            initial_request=request if request.request_id in initial else None,
                        )
                        # Count completed grading/checkpoint writes, including recorded errors.
                        progress.update(1)

    def _summary(self, status: str) -> dict:
        return {
            "run_id": self.run_id,
            "status": status,
            "output_dir": str(self.output_dir),
            "total": len(self.states),
            "finished": sum(state.phase == "done" for state in self.states.values()),
            "solved": sum(state.solved for state in self.states.values()),
            "errors": sum(len(state.errors) for state in self.states.values()),
            "initial_response_cohort_fingerprint": fingerprint([
                [q_id, state.initial_response_fingerprint]
                for q_id, state in sorted(self.states.items())
            ]),
            "stop_reasons": dict(
                Counter(state.stop_reason for state in self.states.values() if state.stop_reason)
            ),
        }

    def run(self) -> dict:
        self._load_dataset()
        if self.verifier is None:
            from utils.util_for_math_verify import MathVerifier

            self.verifier = MathVerifier()
        if hasattr(self.verifier, "prepare"):
            for answer in dict.fromkeys(problem["answer"] for problem in self.problems.values()):
                self.verifier.prepare(answer)
        identity = {
            "schema_version": SCHEMA_VERSION,
            "models": {role: asdict(spec) for role, spec in self.specs.items()},
            "prompts": self.config["prompts"],
            "seed": self.args["seed"],
            "verifier": get_verifier_identity(self.verifier),
            "prompt_rendering_version": PROMPT_RENDERING_VERSION,
            "initial_response_mode": self.args["initial_response_mode"],
            "max_feedback_count": self.args["max_feedback_count"],
            "collect_final_teacher_response": self.args["collect_final_teacher_response"],
            "dataset_fingerprint": fingerprint(
                [
                    {key: problem[key] for key in ("id", "question", "answer")}
                    for problem in self.problems.values()
                ]
            ),
        }
        if self.args["initial_response_mode"] != "off":
            self.initial_cache = InitialResponseCache(
                Path(self.config["paths"]["output_dir"]) / "initial_student_responses",
                initial_response_identity(
                    self.specs["student"], dataset_name=self.args["target_data"],
                    instruction=self.config["prompts"]["student"], seed=self.args["seed"],
                    verifier=identity["verifier"],
                    prompt_rendering_version=PROMPT_RENDERING_VERSION,
                ),
            )
        self.store = RunStore(self.output_dir, identity, self.config, self.resume)
        if self.manager is None:
            self.manager = ModelManager(self.specs, self.output_dir)
        logger.info("Run directory: %s", self.output_dir)
        try:
            with self.manager:
                for q_id, problem in self.problems.items():
                    state = self.store.load(q_id) if self.resume else None
                    self.states[q_id] = state or DialogueState(
                        q_id,
                        [{"role": "teacher", "content": get_start_utterance(problem["question"])}],
                    )
                    self.store.save(self.states[q_id])
                self._recover_dispatches()
                while any(state.phase != "done" for state in self.states.values()):
                    # Alternate dialogue turns; both roles keep their GPU replicas loaded.
                    self._run_role("student")
                    self._run_role("teacher")
                    write_json(self._summary("running"), self.output_dir / "summary.json")
            summary = self._summary("completed")
            if summary["errors"]:
                summary["status"] = "completed_with_errors"
            write_json(summary, self.output_dir / "summary.json")
            return summary
        except BaseException:
            try:
                write_json(self._summary("interrupted"), self.output_dir / "summary.json")
            except Exception:
                logger.exception("Could not save the interruption summary")
            logger.error("Run interrupted. Resume with run.id=%s run.resume=true", self.run_id)
            raise
        finally:
            self.store.close()


def run_from_config(cfg: DictConfig) -> dict | None:
    config = prepare_hydra_config(cfg)
    options = config["run"]
    simulator = TutoringSimulator(config, run_id=options["id"], resume=options["resume"])
    if options["check_config"]:
        for role, spec in simulator.specs.items():
            print(f"{role}: {spec.backend} / {spec.model}")
            if spec.backend == "vllm":
                print(
                    f"  GPUs: {spec.options['cuda_visible_devices']}; "
                    f"data_parallel_size: {spec.options['data_parallel_size']}; "
                    f"tensor_parallel_size per replica: {spec.options['engine_kwargs']['tensor_parallel_size']}"
                )
        print(f"max_feedback_count: {simulator.args['max_feedback_count']}")
        print(f"run.id: {options['id'] or '(automatic)'}; resume: {options['resume']}")
        print(f"output: {simulator.output_dir}")
        return None
    summary = simulator.run()
    print(f"Solved {summary['solved']}/{summary['total']}; errors={summary['errors']}")
    print(f"Results: {summary['output_dir']}")
    if summary["errors"]:
        # Hydra treats ordinary exceptions as failed jobs, including in multirun.
        raise RuntimeError(f"Simulation completed with {summary['errors']} errors; see results")
    return summary


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_from_config(cfg)


if __name__ == "__main__":

    def stop_on_signal(signum, frame):
        # Let the context managers close the local worker on scheduler termination.
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_signal)
    try:
        main()
    except KeyboardInterrupt:
        logger.error("Stopped by user")
        raise SystemExit(130) from None
