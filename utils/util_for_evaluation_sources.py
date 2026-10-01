"""Select complete simulator runs by explicit paths or model combinations."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from .util import read_json
from .util_for_evaluation_io import SourceRun, load_source

logger = logging.getLogger(__name__)
_INVALID_SOURCE_ERRORS = (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError, OverflowError)


def _combinations(config: dict) -> list[tuple[str, str, str]]:
    evaluation = config["evaluation"]
    teachers = evaluation.get("teacher_model_names", [])
    students = evaluation.get("student_model_names", [])
    target_data = evaluation.get("target_data")
    if not teachers or not students or not isinstance(target_data, str) or not target_data:
        raise ValueError(
            "Automatic source selection requires evaluation.teacher_model_names, "
            "evaluation.student_model_names and evaluation.target_data"
        )
    return [(teacher, student, target_data) for teacher in teachers for student in students]


def _source_combination(source: SourceRun) -> tuple[str, str, str]:
    return source.teacher, source.student, source.target_data


def _created_at(path: Path, combination: tuple[str, str, str]) -> datetime:
    manifest = read_json(path / "manifest.json")
    actual = (
        manifest["identity"]["models"]["teacher"]["name"],
        manifest["identity"]["models"]["student"]["name"],
        manifest["config"]["test_args"]["target_data"],
    )
    if actual != combination:
        raise ValueError(f"Directory combination {combination} does not match manifest source {actual}")
    value = manifest.get("created_at")
    if not isinstance(value, str):
        raise ValueError("manifest.created_at must be a timezone-aware ISO timestamp")
    try:
        created = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("manifest.created_at must be a timezone-aware ISO timestamp") from exc
    if created.tzinfo is None or created.utcoffset() is None:
        raise ValueError("manifest.created_at must include a timezone")
    return created.astimezone(UTC)


def _pinned_sources(
    config: dict, output_dir: Path, combinations: list[tuple[str, str, str]],
    dataset_loader: Callable | None,
) -> list[SourceRun]:
    manifest_path = output_dir / "manifest.json"
    try:
        manifest = read_json(manifest_path)
        paths = manifest["config"]["evaluation"]["run_dirs"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Cannot resume automatic source selection: no saved run_dirs in {manifest_path}") from exc
    if not isinstance(paths, list) or not paths or any(
        not isinstance(path, str) or not path.strip() or not Path(path).is_absolute() for path in paths
    ):
        raise ValueError(f"Cannot resume automatic source selection: invalid saved run_dirs in {manifest_path}")
    sources = []
    for path in paths:
        try:
            source = load_source(path, config, dataset_loader)
        except _INVALID_SOURCE_ERRORS as exc:
            raise ValueError(f"Cannot resume pinned simulator source {path}: {exc}") from exc
        sources.append(source)
    actual = [_source_combination(source) for source in sources]
    expected = set(combinations)
    if len(actual) != len(expected) or set(actual) != expected:
        missing = sorted(expected - set(actual))
        unexpected = sorted(set(actual) - expected)
        raise ValueError(
            "Automatic resume selectors do not match the saved source combinations: "
            f"missing={missing}, unexpected={unexpected}, saved_count={len(actual)}. "
            "Start a new evaluation run to change teacher/student/dataset selection."
        )
    by_combination = {_source_combination(source): source for source in sources}
    selected = [by_combination[combination] for combination in combinations]
    for source in selected:
        logger.info("Resuming with pinned simulator source: %s", source.path)
    return selected


def select_sources(
    config: dict, *, output_dir: Path, dataset_loader: Callable | None = None
) -> list[SourceRun]:
    """Return validated snapshots without changing config or source directories.

    Explicit ``run_dirs`` keep their existing behavior. Automatic selection uses
    each teacher/student/dataset directory's newest valid, complete run by its
    timezone-aware manifest ``created_at``. Equal timestamps choose the larger
    run-directory name lexicographically. Summary files and filesystem mtimes
    never determine completion or ordering. Automatic resume uses only the
    evaluation manifest's saved paths, even when newer simulations now exist.
    """
    paths = config["evaluation"].get("run_dirs", [])
    if paths:
        sources = [load_source(path, config, dataset_loader) for path in paths]
        for source in sources:
            logger.info("Selected explicit simulator source: %s", source.path)
        return sources

    combinations = _combinations(config)
    if config.get("run", {}).get("resume", False):
        return _pinned_sources(config, Path(output_dir), combinations, dataset_loader)

    simulation_dir = Path(config["paths"]["simulation_dir"])
    selected, failures = [], []
    for combination in combinations:
        teacher, student, target_data = combination
        directory = simulation_dir / teacher / student / target_data
        rejected = []
        ordered = []
        try:
            candidates = sorted((path for path in directory.iterdir() if path.is_dir()), key=lambda path: path.name)
        except OSError as exc:
            candidates = []
            rejected.append(f"{directory}: {type(exc).__name__}: {exc}")
        for path in candidates:
            try:
                ordered.append((_created_at(path, combination), path.name, path))
            except _INVALID_SOURCE_ERRORS as exc:
                reason = f"{path}: {type(exc).__name__}: {exc}"
                rejected.append(reason)
                logger.warning("Skipping simulator source %s", reason)
        chosen = None
        for _, _, path in sorted(ordered, reverse=True):
            try:
                source = load_source(path, config, dataset_loader)
                if _source_combination(source) != combination:
                    raise ValueError(
                        f"Directory combination {combination} does not match manifest source "
                        f"{_source_combination(source)}"
                    )
            except _INVALID_SOURCE_ERRORS as exc:
                reason = f"{path}: {type(exc).__name__}: {exc}"
                rejected.append(reason)
                logger.warning("Skipping simulator source %s", reason)
                continue
            chosen = source
            break
        if chosen is None:
            reasons = "\n    ".join(rejected) if rejected else "No simulator run directories found"
            failures.append(
                f"teacher={teacher}, student={student}, dataset={target_data}\n"
                f"  searched: {directory}\n    {reasons}"
            )
        else:
            logger.info("Selected latest complete simulator source: %s", chosen.path)
            selected.append(chosen)
    if failures:
        raise ValueError("No valid completed simulator run for every requested combination:\n" + "\n".join(failures))
    return selected
