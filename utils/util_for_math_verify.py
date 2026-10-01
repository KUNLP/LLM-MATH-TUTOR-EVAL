"""Lazy symbolic grading with cached, validated reference answers."""

from __future__ import annotations

from typing import Any


class MathVerifier:
    """A reusable verifier. Call ``prepare`` before inference to validate golds."""

    def __init__(self) -> None:
        self._gold_cache: dict[str, list[Any]] = {}
        self._backend: tuple[Any, Any, tuple[Any, ...]] | None = None

    def _load_backend(self) -> tuple[Any, Any, tuple[Any, ...]]:
        if self._backend is None:
            try:
                from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
            except ImportError:
                raise RuntimeError(
                    "Mathematical grading requires the 'math-verify' package"
                ) from None
            self._backend = (parse, verify, (LatexExtractionConfig(), ExprExtractionConfig()))
        return self._backend

    def prepare(self, gold_answer: str) -> None:
        """Validate and cache a gold expression without changing its source text."""
        if not isinstance(gold_answer, str) or not gold_answer.strip():
            raise ValueError("Gold answer must be a nonblank string")
        if gold_answer in self._gold_cache:
            return
        parse, _, config = self._load_backend()
        gold = gold_answer.strip()
        # Raw PRM/MATH LaTeX needs an environment. Wrapping first also avoids
        # extracting just a denominator from an unwrapped fraction as a number.
        if not ("$" in gold or r"\[" in gold or r"\(" in gold or r"\boxed" in gold):
            gold = f"${gold}$"
        try:
            extracted = parse(
                gold, extraction_config=config, fallback_mode="no_fallback", raise_on_error=True
            )
        except Exception:
            raise ValueError("Gold answer could not be parsed for mathematical grading") from None
        if not extracted:
            raise ValueError("Gold answer could not be parsed for mathematical grading")
        self._gold_cache[gold_answer] = extracted

    def __call__(self, output: str, gold_answer: str) -> bool:
        self.prepare(gold_answer)
        if not isinstance(output, str) or not output.strip():
            return False
        parse, verify, config = self._load_backend()
        try:
            prediction = parse(
                output, extraction_config=config, fallback_mode="no_fallback", raise_on_error=True
            )
            if not prediction:
                return False
            return bool(
                verify(
                    self._gold_cache[gold_answer], prediction, float_rounding=6, raise_on_error=True
                )
            )
        except Exception:
            raise RuntimeError("Mathematical grading failed") from None


_default_verifier = MathVerifier()


def verify_correctness(output: str, gold_answer: str) -> bool:
    return _default_verifier(output, gold_answer)
