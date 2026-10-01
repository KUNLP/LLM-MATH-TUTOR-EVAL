"""Evaluate saved tutoring runs with the original experiment's metric policies.

The CLI loads no models, credentials or source data during import/config checks.
Raw simulator runs are immutable; inference caches and annotations live in a
separate resumable evaluation directory.
"""

from __future__ import annotations

import logging
import signal
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from utils.util import fingerprint, write_json
from utils.util_for_config import register_hydra_resolvers, validate_component
from utils.util_for_evaluation_config import (
    normalize_evaluation_config,
    prepare_evaluation_config,
    resolve_checker_spec,
    resolve_student_spec,
)
from utils.util_for_evaluation_io import (
    POLICY_VERSION,
    CachedEvaluationBackend,
    EvaluationStore,
    validate_comparison_sources,
)
from utils.util_for_evaluation_metrics import evaluate_dialogues
from utils.util_for_evaluation_report import normalize_results, write_reports
from utils.util_for_evaluation_sources import select_sources
from utils.util_for_evaluation_checkers import (
    OVERINFORMATIVE_INSTRUCTION, TEACHER_JUDGMENT_CLASSIFICATION_INSTRUCTION,
)
from utils.util_for_policies import PROMPT_RENDERING_VERSION, get_verifier_identity

logger = logging.getLogger(__name__)
register_hydra_resolvers()


def _local_backend(spec):
    from utils.util_for_dp import DataParallelBackend
    from utils.util_for_evaluation_backend import EvaluationVLLMBackend

    return DataParallelBackend(spec, factory=EvaluationVLLMBackend)


class SimulationAnalyzer:
    def __init__(
        self,
        config: dict,
        *,
        dataset_loader: Callable | None = None,
        backend_factory: Callable | None = None,
        verifier: Callable | None = None,
    ):
        self.config = normalize_evaluation_config(config)
        self.args = self.config["evaluation"]
        self.dataset_loader = dataset_loader
        self.backend_factory = backend_factory or _local_backend
        self.verifier = verifier
        self.run_id = validate_component(
            self.config["run"]["id"]
            or datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8],
            "run.id",
        )
        self.output_dir = Path(self.config["paths"]["output_dir"]) / self.run_id

    def _identity(self, sources, checker_spec) -> dict:
        return {
            "policy_version": POLICY_VERSION,
            "verifier": get_verifier_identity(self.verifier),
            "prompt_rendering_version": PROMPT_RENDERING_VERSION,
            "evaluation_prompts": {
                "overinformative": OVERINFORMATIVE_INSTRUCTION,
                "teacher_judgment": TEACHER_JUDGMENT_CLASSIFICATION_INSTRUCTION,
            },
            "source_verifiers": {source.key: source.identity.get("verifier") for source in sources},
            "comparison_mode": self.args["comparison_mode"],
            "sources": {source.key: source.digest for source in sources},
            "students": {
                source.key: asdict(resolve_student_spec(self.config, source.identity))
                for source in sources if source.identity["models"]["student"]["backend"] == "vllm"
            },
            "student_options": self.args["student_options"],
            "vllm": self.config.get("vllm", {}),
            "checker": asdict(checker_spec),
            "seed": self.args["seed"],
            "num_prev_steps": self.args["num_prev_steps"],
        }

    def _checkers(self, source):
        from utils.util_for_evaluation_checkers import EvaluationCheckers

        return EvaluationCheckers(
            seed=self.args["seed"], batch_size=self.args["batch_size"],
            num_prev_steps=self.args["num_prev_steps"],
            student_instruction=source.identity["prompts"]["student"], verifier=self.verifier,
        )

    def _student_stages(self, sources, store) -> None:
        stage = self.args["stage"]
        groups = defaultdict(list)
        specs = {}
        cache_specs = store.saved_identity["students"]
        for source in sources:
            spec = resolve_student_spec(self.config, source.identity)
            cache_spec = cache_specs[source.key]
            # Original namespaces can differ only in old timeout settings even
            # when today's runtime specs match. Do not read a sibling's cache.
            key = (fingerprint(asdict(spec)), fingerprint(cache_spec))
            groups[key].append(source)
            specs[key] = (spec, cache_spec)
        for key, group in groups.items():
            spec, cache_spec = specs[key]
            logger.info("Student evaluation: %s (%d simulator runs)", spec.name, len(group))
            with CachedEvaluationBackend(
                spec, store.path / "_responses", self.backend_factory, cache_spec=cache_spec,
            ) as backend:
                for source in tqdm(group, desc=f"Evaluate {spec.name}"):
                    checkers = self._checkers(source)
                    annotated = store.stage(source, "overinformative")
                    if stage in {"all", "overinformative"} and annotated is None:
                        annotated = checkers.annotate_overinformative(
                            source.dialogues, source.problems, backend
                        )
                        store.save_stage(source, "overinformative", annotated)
                    if stage in {"all", "answer_reachability_gain"}:
                        if annotated is None:
                            raise ValueError("Run the overinformative stage first in this evaluation run")
                        if store.stage(source, "answer_reachability_gain") is None:
                            annotated = checkers.annotate_reachability(annotated, source.problems, backend)
                            store.save_stage(source, "answer_reachability_gain", annotated)

    def _judgment_stage(self, sources, store, checker_spec) -> None:
        with CachedEvaluationBackend(
            checker_spec, store.path / "_responses", self.backend_factory,
            cache_spec=store.saved_identity["checker"],
        ) as backend:
            for source in tqdm(sources, desc=f"Teacher judgments ({checker_spec.name})"):
                if store.stage(source, "teacher_judgment") is not None:
                    continue
                annotated = store.stage(source, "answer_reachability_gain")
                if annotated is None:
                    raise ValueError("Run the answer_reachability_gain stage first in this evaluation run")
                annotated = self._checkers(source).annotate_teacher_judgment(annotated, backend)
                store.save_stage(source, "teacher_judgment", annotated)

    def _report(self, sources, store, checker_spec) -> list[dict]:
        rows = []
        for source in sources:
            annotated = store.stage(source, "teacher_judgment")
            if annotated is None:
                raise ValueError("Run the teacher_judgment stage before generating reports")
            metrics = evaluate_dialogues(annotated, source.identity["max_feedback_count"])
            errors = Counter(
                error.get("stage", "unknown")
                for dialogue in annotated.values() for utterance in dialogue
                for error in utterance.get("annotation_errors", [])
            )
            rows.append({
                "teacher_model_name": source.teacher, "student_model_name": source.student,
                "checker_model_name": checker_spec.name, "target_data": source.target_data,
                "source_run": str(source.path),
                "source_verifier_status": "recorded" if source.identity.get("verifier") is not None else "legacy_unknown",
                "source_renderer_status": "recorded" if source.identity.get("prompt_rendering_version") is not None else "legacy_unknown",
                "max_feedback_count": source.identity["max_feedback_count"],
                **metrics,
                "annotation_error_counts": dict(errors),
                "simulation_stop_reasons": dict(Counter(state["stop_reason"] for state in source.states.values())),
            })
        rows = normalize_results(rows)
        write_reports(store.path, rows, write_excel=self.args["write_excel"])
        return rows

    def run(self) -> dict:
        sources = select_sources(
            self.config, output_dir=self.output_dir, dataset_loader=self.dataset_loader,
        )
        validate_comparison_sources(sources, comparison_mode=self.args["comparison_mode"])
        verifier_identity = get_verifier_identity(self.verifier)
        for source in sources:
            recorded_verifier = source.identity.get("verifier")
            if recorded_verifier is None:
                logger.warning("Source %s has no verifier policy metadata; retaining recorded grades (legacy_unknown)", source.path)
            elif recorded_verifier != verifier_identity:
                raise ValueError(f"Source verifier policy differs from evaluation verifier: {source.path}; "
                                 "use the matching verifier rather than mixing grading policies")
            recorded_renderer = source.identity.get("prompt_rendering_version")
            if recorded_renderer is None:
                logger.warning("Source %s has no prompt-rendering policy metadata (legacy_unknown)", source.path)
            elif recorded_renderer != PROMPT_RENDERING_VERSION:
                raise ValueError(f"Source prompt-rendering policy differs from evaluation renderer: {source.path}")
        for source in sources:
            if self.output_dir == source.path or self.output_dir.is_relative_to(source.path):
                raise ValueError("Evaluation output must not be inside a source simulator run")
        # Persist the exact selection so resume remains stable when new
        # simulations appear under the discovery root.
        self.args["run_dirs"] = [str(source.path) for source in sources]
        checker_spec = resolve_checker_spec(self.config)
        if self.args["stage"] in {"all", "overinformative", "answer_reachability_gain"}:
            for source in sources:
                resolve_student_spec(self.config, source.identity)
        store = EvaluationStore(
            self.output_dir, self._identity(sources, checker_spec), self.config,
            self.config["run"]["resume"],
        )
        summary = {"run_id": self.run_id, "stage": self.args["stage"], "source_count": len(sources),
                   "output_dir": str(self.output_dir), "status": "running"}
        try:
            if store.prepare_retry(sources, requested=self.args["retry_failed"]):
                # A repair includes affected downstream annotations and reports.
                self.args["stage"] = "all"
                summary["stage"] = "all"
            write_json(summary, store.path / "summary.json")
            stage = self.args["stage"]
            if stage in {"all", "overinformative", "answer_reachability_gain"}:
                self._student_stages(sources, store)
            if stage in {"all", "teacher_judgment"}:
                self._judgment_stage(sources, store, checker_spec)
            if stage in {"all", "report"}:
                rows = self._report(sources, store, checker_spec)
            summary["annotation_errors"] = store.annotation_error_count(sources)
            summary["legacy_unknown_verifier_sources"] = sum(source.identity.get("verifier") is None for source in sources)
            summary["legacy_unknown_renderer_sources"] = sum(source.identity.get("prompt_rendering_version") is None for source in sources)
            if summary["annotation_errors"]:
                logger.warning("Evaluation contains %d annotation errors; resume with evaluation.retry_failed=true to repair them",
                               summary["annotation_errors"])
            summary["status"] = "completed_with_errors" if summary["annotation_errors"] else "completed"
            write_json(summary, store.path / "summary.json")
            store.finish_retry()
            return summary
        except BaseException:
            summary["status"] = "interrupted"
            try:
                write_json(summary, store.path / "summary.json")
            except OSError:
                logger.exception("Could not save the interruption summary")
            logger.error("Evaluation interrupted; resume with run.id=%s run.resume=true", self.run_id)
            raise
        finally:
            store.close()


def run_from_config(cfg: DictConfig) -> dict | None:
    config = prepare_evaluation_config(cfg)
    analyzer = SimulationAnalyzer(config)
    if config["run"]["check_config"]:
        checker = resolve_checker_spec(config)
        print(f"stage: {analyzer.args['stage']}")
        if analyzer.args["run_dirs"]:
            print(f"explicit source runs: {len(analyzer.args['run_dirs'])}")
        else:
            print(f"automatic sources: {analyzer.args['teacher_model_names']} x "
                  f"{analyzer.args['student_model_names']} / {analyzer.args['target_data']}")
            print(f"simulation directory: {config['paths']['simulation_dir']}")
        print("Source completeness is checked before inference when evaluation runs.")
        print(f"checker: {checker.name} / {checker.model}")
        print(f"output: {analyzer.output_dir}")
        return None
    result = analyzer.run()
    print(f"Evaluation {result['status']}; source runs: {result['source_count']}")
    print(f"Results: {result['output_dir']}")
    return result


@hydra.main(version_base="1.3", config_path="configs", config_name="analyze")
def main(cfg: DictConfig) -> None:
    run_from_config(cfg)


if __name__ == "__main__":
    def stop_on_signal(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_signal)
    try:
        main()
    except KeyboardInterrupt:
        logger.error("Stopped by user")
        raise SystemExit(130) from None
