"""Compose experiment settings and resolve explicit per-role backends."""

from __future__ import annotations

import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .util_for_instruction import STUDENT_INSTRUCTION, TEACHER_INSTRUCTION
from .util_for_types import ModelSpec

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _project_path(root: str, path: str) -> str:
    base = PROJECT_ROOT / Path(root).expanduser()
    return str((base / Path(path).expanduser()).resolve())


def register_hydra_resolvers() -> None:
    if not OmegaConf.has_resolver("project_root"):
        OmegaConf.register_new_resolver("project_root", lambda: str(PROJECT_ROOT))
    if not OmegaConf.has_resolver("project_path"):
        OmegaConf.register_new_resolver("project_path", _project_path)


def prepare_hydra_config(cfg: DictConfig) -> dict:
    """Resolve the composed YAML before it crosses into the simulation engine."""
    raw = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(raw, dict):
        raise ValueError("The experiment configuration must be a mapping")
    paths = raw.get("paths", {})
    root = _project_path(str(PROJECT_ROOT), paths.get("root_dir") or str(PROJECT_ROOT))
    return normalize_config(raw, root)


def load_yaml(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML support requires PyYAML: pip install -e .") from exc
    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return data


def _positive_int(value: Any, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _path(value: str | Path, root: Path) -> str:
    path = Path(value).expanduser()
    return str((root / path).resolve() if not path.is_absolute() else path.resolve())


def validate_component(value: str, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{label} must be a simple name using letters, digits, dots, '_' or '-'")
    return value


def _merge_settings(*layers: dict, label: str) -> dict:
    """Apply later leaves without losing other nested settings or mutating inputs."""
    merged = {}
    for layer in layers:
        if not isinstance(layer, dict):
            raise ValueError(f"{label} must be a mapping")
        for key, value in layer.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = _merge_settings(merged[key], value, label=f"{label}.{key}")
            else:
                merged[key] = deepcopy(value)
    return merged


def _cuda_devices(value: Any, label: str) -> str | None:
    """Normalize an explicit physical GPU selection without loading CUDA."""
    if value is None:
        return None
    if isinstance(value, str):
        devices = [device.strip() for device in value.split(",")]
        for index, device in enumerate(devices):
            if re.fullmatch(r"[0-9]+", device):
                devices[index] = str(int(device))
            elif not re.fullmatch(r"(?:GPU|MIG)-[A-Za-z0-9_./-]+", device):
                raise ValueError(f"{label} needs non-negative GPU indices or GPU/MIG UUIDs")
    elif isinstance(value, list):
        if any(type(device) is not int or device < 0 for device in value):
            raise ValueError(f"{label} must contain non-negative integer GPU indices")
        devices = [str(device) for device in value]
    else:
        raise ValueError(f"{label} must be a comma-separated string, integer list, or null")
    if not devices or len(set(devices)) != len(devices):
        raise ValueError(f"{label} must select at least one GPU without duplicates")
    return ",".join(devices)


def _local_gpu_settings(options: dict, role: str) -> tuple[str, int, dict]:
    """Resolve independent single-GPU engines without initializing CUDA."""
    devices = _cuda_devices(
        options.get("cuda_visible_devices"), f"{role}.options.cuda_visible_devices"
    )
    if devices is None:
        raise ValueError(
            f"{role}.options.cuda_visible_devices must explicitly select at least one GPU"
        )
    device_count = len(devices.split(","))
    replica_count = options.get("data_parallel_size", "auto")
    if replica_count == "auto":
        replica_count = device_count
    _positive_int(replica_count, f"{role}.options.data_parallel_size")
    if replica_count != device_count:
        raise ValueError(
            f"{role}.options.data_parallel_size ({replica_count}) must equal "
            f"the selected GPU count ({device_count})"
        )
    engine = _merge_settings(options.get("engine_kwargs", {}), label=f"{role}.engine_kwargs")
    for key in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"):
        value = engine.get(key, 1)
        if key == "tensor_parallel_size" and value == "auto":
            value = 1
        _positive_int(value, f"{role}.engine_kwargs.{key}")
        if value != 1:
            raise ValueError(
                f"{role}.engine_kwargs.{key} must be 1 for DP: "
                "each selected GPU runs a complete model replica"
            )
        if key == "tensor_parallel_size" or key in engine:
            engine[key] = value
    engine.setdefault("gpu_memory_utilization", 0.9)
    return devices, device_count, engine


def validate_model_allocations(specs: dict[str, ModelSpec]) -> None:
    """Require disjoint explicit local allocations; do not query GPU hardware.

    CUDA accepts indices and UUID prefixes. Mixing those selector forms makes
    overlap impossible to check without hardware discovery, so local roles must
    use a consistent form. MIG selectors identify instances, whose physical-card
    ownership cannot be verified here. Device existence is checked at startup.
    """
    allocated: list[tuple[str, str]] = []
    selector_form = None
    for role, spec in specs.items():
        if spec.backend != "vllm":
            continue
        devices, _, _ = _local_gpu_settings(spec.options, role)
        for device in devices.split(","):
            form = "index" if device.isdecimal() else device.split("-", 1)[0]
            if selector_form is not None and form != selector_form:
                raise ValueError(
                    "Local GPU allocations must consistently use indices, GPU UUIDs, or "
                    "MIG UUIDs; mixing selector forms cannot verify disjoint GPUs"
                )
            selector_form = form
            for owner, other in allocated:
                same_device = device == other
                if form != "index":
                    same_device = same_device or device.startswith(other) or other.startswith(device)
                if same_device:
                    raise ValueError(
                        f"GPU {device!r} overlaps allocations for {owner} and {role}; "
                        "each local model replica requires a separate GPU"
                    )
            allocated.append((role, device))


def normalize_config(raw: dict, root_dir: str | Path) -> dict:
    """Return a serializable configuration without reading credentials or data."""
    config = deepcopy(raw)
    root = Path(root_dir).resolve()
    paths = config.setdefault("paths", {})
    for name, default in (("input_dir", "input"), ("output_dir", "output"), ("cache_dir", "cache")):
        paths[name] = _path(paths.get(name) or default, root)
    paths["root_dir"] = str(root)
    for name in ("huggingface_access_token_file", "api_key_file", "dataset_file"):
        if paths.get(name):
            paths[name] = _path(paths[name], root)

    run = config.setdefault("run", {})
    if not isinstance(run, dict):
        raise ValueError("run must be a mapping")
    unknown = set(run) - {"id", "resume", "check_config"}
    if unknown:
        raise ValueError(f"Unknown run settings: {', '.join(sorted(unknown))}")
    run.setdefault("id", None)
    run.setdefault("resume", False)
    run.setdefault("check_config", False)
    if run["id"] is not None:
        validate_component(run["id"], "run.id")
    for name in ("resume", "check_config"):
        if type(run[name]) is not bool:
            raise ValueError(f"run.{name} must be a boolean")
    if run["resume"] and run["id"] is None:
        raise ValueError("run.resume=true requires an explicit run.id")

    args = config.setdefault("test_args", {})
    args.setdefault("seed", 1234)
    _positive_int(args["seed"], "seed", 0)
    args.setdefault("target_data", "math_prm800k")
    validate_component(args["target_data"], "target_data")
    if "max_turns" in args:
        turns = _positive_int(args.pop("max_turns"), "max_turns", 2)
        legacy_feedback_count = max(0, (turns - 2) // 2)
        if "max_feedback_count" in args and args["max_feedback_count"] != legacy_feedback_count:
            raise ValueError("Conflicting max_turns and max_feedback_count; specify one budget")
        args["max_feedback_count"] = legacy_feedback_count
    else:
        args.setdefault("max_feedback_count", 10)
    if "make_initial_dialogue_state" in args:
        legacy_initial = args.pop("make_initial_dialogue_state")
        if type(legacy_initial) is not bool:
            raise ValueError("make_initial_dialogue_state must be a boolean")
        legacy_mode = "refresh" if legacy_initial else "reuse"
        if "initial_response_mode" in args and args["initial_response_mode"] != legacy_mode:
            raise ValueError("Conflicting make_initial_dialogue_state and initial_response_mode")
        args["initial_response_mode"] = legacy_mode
    args.setdefault("initial_response_mode", "off")
    if not isinstance(args["initial_response_mode"], str) or args["initial_response_mode"] not in {"off", "reuse", "refresh"}:
        raise ValueError("initial_response_mode must be off, reuse, or refresh")
    _positive_int(args["max_feedback_count"], "max_feedback_count", 0)
    args.setdefault("batch_size", 64)
    _positive_int(args["batch_size"], "batch_size")
    args.setdefault("debug", False)
    args.setdefault("collect_final_teacher_response", True)
    args.setdefault("offline", False)
    for name in ("debug", "collect_final_teacher_response", "offline"):
        if type(args[name]) is not bool:
            raise ValueError(f"{name} must be a boolean")
    if args.get("limit") is not None:
        _positive_int(args["limit"], "limit")
    config.setdefault("prompts", {})
    specs = {}
    for role, default in (("student", STUDENT_INSTRUCTION), ("teacher", TEACHER_INSTRUCTION)):
        config["prompts"].setdefault(role, default)
        if not isinstance(config["prompts"][role], str) or not config["prompts"][role].strip():
            raise ValueError(f"prompts.{role} must be a non-empty string")
        specs[role] = resolve_model_spec(config, role)
    validate_model_allocations(specs)
    return config


def resolve_model_spec(config: dict, role: str) -> ModelSpec:
    selection = config["test_args"].get(role)
    if isinstance(selection, str):
        name, override = selection, {}
    elif isinstance(selection, dict) and isinstance(selection.get("name"), str):
        override = deepcopy(selection)
        name = override.pop("name")
    else:
        raise ValueError(f"test_args.{role} must be a model name or a mapping with 'name'")
    validate_component(name, f"{role} model name")
    base = deepcopy(config.get("models", {}).get(name, {}))
    if not base and "model" not in override:
        raise ValueError(f"Model {name!r} is not defined in model_list.yaml")
    if "api_key" in base or "api_key" in override:
        raise ValueError("Use api_key_env or api_key_file instead of a literal API key")
    unknown = (set(base) | set(override)) - {"model", "backend", "generation", "options"}
    if unknown:
        raise ValueError(f"Unsupported model settings for {name}: {', '.join(sorted(unknown))}")
    model_options = base.get("options", {})
    role_options = override.get("options", {})
    base["generation"] = _merge_settings(
        base.get("generation", {}), override.get("generation", {}), label=f"{name}.generation"
    )
    base.update(
        {key: value for key, value in override.items() if key not in {"generation", "options"}}
    )
    backend = base.get("backend", "vllm")
    if backend not in {"vllm", "api"}:
        raise ValueError(f"Invalid backend for {name}: {backend}")
    model = base.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"Missing model identifier for {name}")
    generation = {"max_tokens": 1000, "temperature": 0.0, **base["generation"]}
    _positive_int(generation["max_tokens"], f"{name}.generation.max_tokens")
    if generation.get("n", 1) != 1:
        raise ValueError("The dialogue simulator requires generation.n == 1")
    for key in ("temperature", "top_p"):
        value = generation.get(key)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"generation.{key} must be finite or null")
    if generation.get("temperature") is not None and generation["temperature"] < 0:
        raise ValueError("generation.temperature must be >= 0")
    if generation.get("top_p") is not None and not 0 < generation["top_p"] <= 1:
        raise ValueError("generation.top_p must be in (0, 1]")
    options = _merge_settings(
        config.get(backend, {}), model_options, role_options, label=f"{name}.options"
    )
    paths = config["paths"]
    if backend == "vllm":
        options.setdefault("cache_dir", paths["cache_dir"])
        options.setdefault("offline", config["test_args"].get("offline", False))
        options.setdefault("hf_token_file", paths.get("huggingface_access_token_file"))
        devices, device_count, engine = _local_gpu_settings(options, role)
        options["cuda_visible_devices"] = devices
        options["data_parallel_size"] = device_count
        options["engine_kwargs"] = engine
    else:
        options.setdefault("api_key_file", paths.get("api_key_file"))
        # Credentials are resolved inside the backend; never serialize literal secrets.
        if "api_key" in options:
            raise ValueError("Use api_key_env or api_key_file instead of a literal API key")
    for key in ("api_key_file", "hf_token_file", "cache_dir"):
        if options.get(key):
            options[key] = _path(options[key], Path(paths["root_dir"]))
    return ModelSpec(
        name=name, backend=backend, model=model, generation=generation, options=options
    )


def load_config(
    root_dir: str | Path,
    config_file_name: str = "config_for_tutoring_simulator.yaml",
    model_list_file_name: str = "model_list.yaml",
) -> dict:
    """Load plain legacy YAML or compose a Hydra YAML's Defaults List.

    Programmatic callers retain the explicit root_dir path convention. CLI
    experiments use prepare_hydra_config and their configured paths.root_dir.
    Hydra registries are selected through defaults; the legacy registry argument
    is only used when loading plain YAML without a Defaults List.
    """
    root = Path(root_dir).resolve()
    config_path = (root / config_file_name).resolve()
    config = load_yaml(config_path)
    if "defaults" in config:
        from hydra import compose, initialize_config_dir

        if (root / model_list_file_name).resolve() != (root / "model_list.yaml").resolve():
            raise ValueError(
                "Hydra configurations select model registries through defaults; "
                "model_list_file_name is only supported for plain YAML"
            )
        register_hydra_resolvers()
        with initialize_config_dir(version_base="1.3", config_dir=str(config_path.parent)):
            composed = compose(config_name=config_path.stem)
            config = OmegaConf.to_container(composed, resolve=True, throw_on_missing=True)
        return normalize_config(config, root)
    models_file = root / model_list_file_name
    registered = load_yaml(models_file).get("models", {}) if models_file.exists() else {}
    config["models"] = {**registered, **config.get("models", {})}
    return normalize_config(config, root)
