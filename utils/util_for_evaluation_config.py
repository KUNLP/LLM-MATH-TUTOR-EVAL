"""Configuration for evaluating saved simulator runs, without loading models."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .util_for_config import PROJECT_ROOT, _merge_settings, resolve_model_spec, validate_component
from .util_for_types import ModelSpec

STAGES = {"all", "overinformative", "answer_reachability_gain", "teacher_judgment", "report"}

# Allocation and process settings may change without selecting different weights
# or changing how the student's conversation is represented.
EXECUTION_OPTIONS = {
    "cuda_visible_devices", "data_parallel_size", "startup_timeout", "request_timeout",
    "shutdown_timeout", "cache_dir", "offline", "hf_token_file",
    "api_key_file", "api_key_env", "timeout", "max_retries",
}
EXECUTION_ENGINE_OPTIONS = {
    "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
    "gpu_memory_utilization", "max_model_len", "max_num_seqs", "max_num_batched_tokens",
    "swap_space", "cpu_offload_gb", "enforce_eager", "disable_log_stats",
}


def semantic_model_options(options: dict) -> dict:
    """Keep checkpoint, tokenizer and prompt options when comparing models."""
    semantic = {key: deepcopy(value) for key, value in options.items()
                if key not in EXECUTION_OPTIONS and key != "engine_kwargs"}
    engine = {key: deepcopy(value) for key, value in options.get("engine_kwargs", {}).items()
              if key not in EXECUTION_ENGINE_OPTIONS}
    if engine:
        semantic["engine_kwargs"] = engine
    return semantic


def normalize_evaluation_config(raw: dict, root_dir: str | Path | None = None) -> dict:
    config = deepcopy(raw)
    paths = config.setdefault("paths", {})
    root = (PROJECT_ROOT / Path(root_dir or paths.get("root_dir") or PROJECT_ROOT)).resolve()
    paths["root_dir"] = str(root)
    for key, default in (
        ("output_dir", "output/from_analyze_simulation_result"),
        ("simulation_dir", "output/from_tutoring_simulator"),
        ("cache_dir", "/home/cache_dir"),
        ("input_dir", "input"),
    ):
        paths[key] = str((root / Path(paths.get(key) or default).expanduser()).resolve())
    for key in ("huggingface_access_token_file", "api_key_file", "dataset_file"):
        if paths.get(key):
            paths[key] = str((root / Path(paths[key]).expanduser()).resolve())
    run = config.setdefault("run", {})
    if not isinstance(run, dict) or set(run) - {"id", "resume", "check_config"}:
        raise ValueError("run accepts only id, resume and check_config")
    run.setdefault("id", None)
    run.setdefault("resume", False)
    run.setdefault("check_config", False)
    if run["id"] is not None:
        validate_component(run["id"], "run.id")
    for key in ("resume", "check_config"):
        if type(run[key]) is not bool:
            raise ValueError(f"run.{key} must be a boolean")
    if run["resume"] and not run["id"]:
        raise ValueError("run.resume=true requires run.id")

    evaluation = config.setdefault("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be a mapping")
    allowed = {
        "run_dirs", "stage", "seed", "batch_size", "num_prev_steps", "checker",
        "student_options", "write_excel", "offline",
        "teacher_model_names", "student_model_names", "target_data", "retry_failed", "comparison_mode",
    }
    if set(evaluation) - allowed:
        raise ValueError(f"Unknown evaluation settings: {sorted(set(evaluation) - allowed)}")
    evaluation.setdefault("run_dirs", [])
    if not isinstance(evaluation["run_dirs"], list) or any(
        not isinstance(path, str) or not path.strip() for path in evaluation["run_dirs"]
    ):
        raise ValueError("evaluation.run_dirs must be a list of simulator run directories")
    evaluation["run_dirs"] = [
        str((root / Path(path).expanduser()).resolve()) for path in evaluation["run_dirs"]
    ]
    if len(set(evaluation["run_dirs"])) != len(evaluation["run_dirs"]):
        raise ValueError("evaluation.run_dirs contains duplicate directories")
    for key in ("teacher_model_names", "student_model_names"):
        names = evaluation.setdefault(key, [])
        if not isinstance(names, list):
            raise ValueError(f"evaluation.{key} must be a list of model names")
        for name in names:
            validate_component(name, f"evaluation.{key}")
        if len(set(names)) != len(names):
            raise ValueError(f"evaluation.{key} contains duplicate model names")
    evaluation.setdefault("target_data", None)
    if evaluation["target_data"] is not None:
        validate_component(evaluation["target_data"], "evaluation.target_data")
    evaluation.setdefault("stage", "all")
    if evaluation["stage"] not in STAGES:
        raise ValueError(f"evaluation.stage must be one of {sorted(STAGES)}")
    for key, default, minimum in (("seed", 1234, 0), ("batch_size", 64, 1), ("num_prev_steps", 1, 1)):
        value = evaluation.setdefault(key, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f"evaluation.{key} must be an integer >= {minimum}")
    for key, default in (("write_excel", True), ("offline", False), ("retry_failed", False)):
        if type(evaluation.setdefault(key, default)) is not bool:
            raise ValueError(f"evaluation.{key} must be a boolean")
    if evaluation["retry_failed"] and not run["resume"]:
        raise ValueError("evaluation.retry_failed=true requires run.resume=true")
    comparison_mode = evaluation.setdefault("comparison_mode", "model")
    if not isinstance(comparison_mode, str) or comparison_mode not in {"model", "unrestricted"}:
        raise ValueError("evaluation.comparison_mode must be model or unrestricted")
    evaluation.setdefault("student_options", {"cuda_visible_devices": "0"})
    if not isinstance(evaluation["student_options"], dict):
        raise ValueError("evaluation.student_options must be a mapping")
    if not isinstance(evaluation["student_options"].get("engine_kwargs", {}), dict):
        raise ValueError("evaluation.student_options.engine_kwargs must be a mapping")
    if semantic_model_options(evaluation["student_options"]):
        raise ValueError("evaluation.student_options may override execution options only; "
                         "checkpoint and prompt options come from the simulation manifest")
    evaluation.setdefault("checker", {"name": "gemma_3_27b_it", "options": {"cuda_visible_devices": "0"}})
    # Validate checker selection even for --check-config, without reading any run.
    resolve_checker_spec(config)
    return config


def prepare_evaluation_config(cfg: DictConfig) -> dict:
    raw = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(raw, dict):
        raise ValueError("Evaluation configuration must be a mapping")
    return normalize_evaluation_config(raw)


def resolve_checker_spec(config: dict) -> ModelSpec:
    settings = dict(config)
    settings["test_args"] = {
        "checker": config["evaluation"]["checker"],
        "offline": config["evaluation"].get("offline", False),
    }
    spec = resolve_model_spec(settings, "checker")
    if spec.backend != "vllm":
        raise ValueError("The original teacher-judgment evaluator requires a local vllm checker")
    return replace(spec, generation={"max_tokens": 1000, "temperature": 0.0})


def resolve_student_spec(config: dict, source_identity: dict) -> ModelSpec:
    """Use the recorded student checkpoint; only execution options may change."""
    recorded = source_identity["models"]["student"]
    selection = {
        "name": recorded["name"],
        "model": recorded["model"],
        "backend": recorded["backend"],
        "generation": {"max_tokens": 1000, "temperature": 0.0},
        "options": _merge_settings(
            semantic_model_options(recorded.get("options", {})),
            config["evaluation"]["student_options"],
            label="evaluation.student_options",
        ),
    }
    settings = dict(config)
    # Avoid today's model registry silently replacing the checkpoint in the run.
    settings["models"] = {}
    settings["test_args"] = {
        "student": selection, "offline": config["evaluation"]["offline"],
    }
    spec = resolve_model_spec(settings, "student")
    if spec.backend != "vllm":
        raise ValueError(
            f"Student {spec.name!r} used an API backend. Original OI/ARG evaluation needs "
            "the student's local weights and prompt log-probabilities; API runs can use "
            "the teacher_judgment stage when prerequisite annotations already exist."
        )
    return spec
