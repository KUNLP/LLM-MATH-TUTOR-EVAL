"""Load math problems without importing inference libraries or changing source IDs."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_HF_DATASETS = {
    "gsm8k": ("openai/gsm8k", "main"),
    "math": ("EleutherAI/hendrycks_math", None),
}
_PRM_FILE = Path("math_problem/MATH_PRM800K_SETTING/test.jsonl")


def _text(row: Mapping[str, Any], field: str, index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Dataset row {index}: {field} must be a nonblank string")
    return value


def _rows(dataset: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows = list(dataset)
    if not rows:
        raise ValueError("Dataset must contain at least one problem")
    for index, row in enumerate(rows, 1):
        if not isinstance(row, Mapping):
            raise ValueError(f"Dataset row {index}: expected an object")
    return rows


def _question_ids(questions: list[str]) -> dict[str, int]:
    if len(questions) != len(set(questions)):
        raise ValueError("Dataset contains duplicate questions")
    return {question: index for index, question in enumerate(sorted(questions))}


def _boxed_answer(solution: str, index: int) -> str:
    """Keep the final boxed answer as LaTeX, including nested braces."""
    matches = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", solution))
    if not matches:
        raise ValueError(f"Dataset row {index}: solution has no boxed answer")
    start = matches[-1].end()
    depth = 1
    for offset in range(start, len(solution)):
        char = solution[offset]
        if char == "{" and (offset == 0 or solution[offset - 1] != "\\"):
            depth += 1
        elif char == "}" and (offset == 0 or solution[offset - 1] != "\\"):
            depth -= 1
            if depth == 0:
                answer = solution[start:offset].strip()
                if answer:
                    return answer
                break
    raise ValueError(f"Dataset row {index}: boxed answer is empty or malformed")


def preprocessing_gsm8k(dataset: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = _rows(dataset)
    questions = [
        re.sub(r"\s+", " ", _text(row, "question", i)).strip() for i, row in enumerate(rows, 1)
    ]
    question_ids = _question_ids(questions)
    results = []
    for index, (row, question) in enumerate(zip(rows, questions), 1):
        original = _text(row, "answer", index)
        solution, separator, answer = original.rpartition("####")
        if not separator or not answer.strip():
            raise ValueError(f"Dataset row {index}: missing GSM8K final answer marker")
        answer = answer.strip().replace(",", "")
        if not answer:
            raise ValueError(f"Dataset row {index}: GSM8K final answer is blank")
        results.append(
            {
                "id": question_ids[question],
                "question": question,
                "answer": answer,
                "solution": re.sub(r"<<[^<>]*>>", "", solution).strip(),
            }
        )
    return sorted(results, key=lambda row: row["id"])


def _preprocessing_math(dataset: Iterable[Mapping[str, Any]], *, prm: bool) -> list[dict[str, Any]]:
    rows = _rows(dataset)
    questions = [_text(row, "problem", i) for i, row in enumerate(rows, 1)]
    question_ids = _question_ids(questions)
    results = []
    for index, (row, question) in enumerate(zip(rows, questions), 1):
        if prm or "answer" in row:
            answer = _text(row, "answer", index)
        else:
            answer = _boxed_answer(_text(row, "solution", index), index)
        result = {"id": question_ids[question], "question": question, "answer": answer}
        if "solution" in row:
            result["solution"] = _text(row, "solution", index)
        results.append(result)
    return results


def preprocessing_math(dataset: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return _preprocessing_math(dataset, prm=False)


def preprocessing_math_prm800k(dataset: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return _preprocessing_math(dataset, prm=True)


def _normalized_rows(dataset: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    results = []
    ids: set[int] = set()
    questions: set[str] = set()
    for index, row in enumerate(_rows(dataset), 1):
        identifier = row.get("id")
        if isinstance(identifier, str) and re.fullmatch(r"[+-]?\d+", identifier.strip()):
            identifier = int(identifier)
        if not isinstance(identifier, int) or isinstance(identifier, bool) or identifier < 0:
            raise ValueError(f"Dataset row {index}: id must be a non-negative integer")
        question = _text(row, "question", index)
        answer = _text(row, "answer", index)
        if identifier in ids:
            raise ValueError(f"Dataset row {index}: duplicate id")
        if question in questions:
            raise ValueError(f"Dataset row {index}: duplicate question")
        ids.add(identifier)
        questions.add(question)
        result = {"id": identifier, "question": question, "answer": answer}
        if "solution" in row:
            result["solution"] = _text(row, "solution", index)
        results.append(result)
    return results


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                raise ValueError(f"Dataset JSONL line {line_number}: invalid JSON") from None
    return rows


def _offline_requested(offline: bool) -> bool:
    return offline or any(
        os.environ.get(key, "").strip().lower() in {"1", "true", "yes", "on"}
        for key in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE")
    )


def _load_dataset_from_cache_or_download(
    name_or_path: str,
    cache_dir: str | Path,
    config_name: str | None = None,
    *,
    offline: bool = False,
) -> Any:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    saved_path = cache_root / f"{name_or_path.replace('/', '___')}_test"
    local_only = _offline_requested(offline)
    if local_only and not saved_path.is_dir():
        raise FileNotFoundError(
            "Offline dataset cache is missing; prepare the saved dataset cache first"
        )
    try:
        import datasets
    except ImportError:
        raise RuntimeError(
            "Loading Hugging Face datasets requires the 'datasets' package"
        ) from None
    if saved_path.is_dir():
        return datasets.load_from_disk(str(saved_path))
    if config_name is not None:
        dataset = datasets.load_dataset(
            name_or_path, name=config_name, cache_dir=str(cache_root), split="test"
        )
    else:
        configs = datasets.get_dataset_config_names(name_or_path, cache_dir=str(cache_root))
        parts = [
            datasets.load_dataset(name_or_path, name=name, cache_dir=str(cache_root), split="test")
            for name in configs
        ]
        dataset = datasets.concatenate_datasets(parts)
    dataset.save_to_disk(str(saved_path))
    return dataset


def get_math_datas(
    name: str,
    cache_dir: str | Path,
    input_dir: str | Path | None = None,
    debug: bool = False,
    *,
    dataset_file: str | Path | None = None,
    limit: int | None = None,
    offline: bool = False,
) -> list[dict[str, Any]]:
    """Return validated problems; assign full-dataset IDs before debug/limit slicing."""
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise ValueError("Dataset limit must be a positive integer")
    preprocessors = {
        "gsm8k": preprocessing_gsm8k,
        "math": preprocessing_math,
        "math_prm800k": preprocessing_math_prm800k,
        "jsonl": _normalized_rows,
    }
    if name not in preprocessors:
        raise ValueError("Unsupported dataset; choose gsm8k, math, math_prm800k, or jsonl")
    if dataset_file is not None:
        raw = _read_jsonl(Path(dataset_file))
    elif name in _HF_DATASETS:
        repository, config = _HF_DATASETS[name]
        raw = _load_dataset_from_cache_or_download(repository, cache_dir, config, offline=offline)
    elif name == "math_prm800k" and input_dir is not None:
        raw = _read_jsonl(Path(input_dir) / _PRM_FILE)
    else:
        raise ValueError("A dataset_file is required, or input_dir for math_prm800k")
    result = preprocessors[name](raw)
    if debug:
        result = result[:100]
    if limit is not None:
        result = result[:limit]
    return result
