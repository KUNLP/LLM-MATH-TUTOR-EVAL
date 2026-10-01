"""Serializable grading and prompt identities without inference imports."""

from __future__ import annotations

import inspect
import json
from importlib import metadata
from typing import Any


PROMPT_RENDERING_VERSION = 2
VERIFIER_POLICY_VERSION = 1
_MATH_VERIFIER_MODULE = "utils.util_for_math_verify"
_MISSING = object()


def _builtin_verifier_identity() -> dict[str, Any]:
    try:
        package_version = metadata.version("math-verify")
    except metadata.PackageNotFoundError:
        package_version = None
    extraction = ["LatexExtractionConfig", "ExprExtractionConfig"]
    return {
        "kind": "math_verify",
        "policy": "validated_latex_and_expression",
        "policy_version": VERIFIER_POLICY_VERSION,
        "math_verify_version": package_version,
        "gold": {
            "extraction": extraction,
            "wrap_unformatted_latex": True,
            "fallback_mode": "no_fallback",
            "raise_on_error": True,
        },
        "prediction": {
            "extraction": list(extraction),
            "fallback_mode": "no_fallback",
            "raise_on_error": True,
        },
        "equivalence": {"float_rounding": 6, "raise_on_error": True},
    }


def get_verifier_identity(verifier: Any = None) -> dict[str, Any]:
    """Describe the grading contract used by source runs and evaluation caches.

    A custom callable can set ``policy_identity`` to a JSON-serializable dict
    when different wrappers implement the same contract. Otherwise its stable
    module and qualified name identify it; changing callable behavior requires
    an explicit policy revision. No object repr or address enters the identity.
    """
    if verifier is None:
        return _builtin_verifier_identity()
    if not callable(verifier):
        raise TypeError("The answer verifier must be callable")
    # getattr_static avoids Mock's synthesized attributes and property side effects.
    explicit = inspect.getattr_static(verifier, "policy_identity", _MISSING)
    if explicit is not _MISSING:
        if not isinstance(explicit, dict):
            raise ValueError("verifier.policy_identity must be a JSON-serializable dict")
        try:
            policy = json.loads(json.dumps(explicit, allow_nan=False, sort_keys=True))
        except (TypeError, ValueError):
            raise ValueError("verifier.policy_identity must be a JSON-serializable dict") from None
        return {"kind": "custom", "policy": policy}
    owner = verifier if inspect.isroutine(verifier) else type(verifier)
    module, name = owner.__module__, owner.__qualname__
    if module == _MATH_VERIFIER_MODULE and name in {"MathVerifier", "verify_correctness"}:
        return _builtin_verifier_identity()
    return {"kind": "custom", "callable": f"{module}.{name}"}
