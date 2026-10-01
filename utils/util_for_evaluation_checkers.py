"""In-memory annotation stages preserving the original evaluation policies."""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import asdict, replace
from collections.abc import Callable, Iterable, Iterator
from itertools import islice
from typing import Any

from .util_for_evaluation_backend import EvaluationRequest, EvaluationResult
from .util_for_instruction import STUDENT_INSTRUCTION
from .util import fingerprint


OVERINFORMATIVE_INSTRUCTION = """Solve the problem using only the given hint.
Do not include any reasoning or steps.
Only output the final numeric answer after \"So, the answer is\""""

TEACHER_JUDGMENT_CLASSIFICATION_INSTRUCTION = """You are a checker determining whether the teacher believes the student has reached the correct answer to the given problem.

Rules:
- Do NOT judge the actual correctness of the student’s answer; only assess the teacher’s stated or implied belief.
- Consider ONLY the teacher’s utterances in the dialogue. Do not use the student’s utterances as evidence.
- Use the most recent explicit judgment by the teacher as the primary basis for your decision. If no explicit judgment exists, use the strongest implied judgment.
- Distinguish verification/doubt from post-endorsement enrichment:
  - If the teacher expresses doubt about correctness or asks to verify correctness (e.g., “are you sure?”, “verify/check/double-check”), treat this as withholding judgment and label NO, unless the teacher has clearly endorsed correctness.
  - If the teacher clearly endorses correctness and then asks for an alternative method, a more efficient solution, a generalization, or a modified condition (e.g., “another way”, “more efficient”, “generalize”, “what if we change…”), this can still be YES.
- If the teacher praises but then corrects or indicates an error (e.g., “but…”, “not quite”, “incorrect”, “there’s a mistake”), label NO.

Label definitions:
- YES: The teacher believes the student has reached the correct answer (explicitly or through a strong implied closure after endorsement).
- NO: All other cases, including explicit rejection, partial acknowledgment, withholding judgment, verification/doubt, or giving only hints/questions without endorsement.

Evidence and output:
- Quote 1–2 exact phrases from the teacher’s utterances that support your decision (verbatim).
- Output the result in the following JSON format:
  {"label":"<YES|NO>", "evidence_spans":["<quote1>","<quote2>"], "confidence":"<high|med|low>"}

Confidence guidance:
- high: explicit endorsement of correctness (e.g., “correct”, “that’s right”, “you got it”, “the answer is …”)
- med: clear endorsement with minor follow-up that does not question correctness (e.g., enrichment)
- low: weak or ambiguous endorsement, mixed signals, or inference based on implication rather than explicit judgment"""


def parse_label(response_text: str) -> str | None:
    """Accept complete JSON or its continuation, then use the original parser."""
    if not isinstance(response_text, str):
        return None
    for candidate in (response_text.strip(), '{"label":' + response_text.strip()):
        try:
            value = json.JSONDecoder().raw_decode(candidate)[0]
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and "label" in value:
            label = value["label"]
            return label.strip().lower() if isinstance(label, str) and label.strip().lower() in {"yes", "no"} else None

    # The original local checker prepends its assistant prefill before parsing.
    compact = re.sub(r"\s+", "", response_text.lower())
    if "label" not in compact:
        compact = '{"label":' + compact
    label_start = compact.find("label")
    tail = compact[label_start + 5:]
    yes_index, no_index = tail.find("yes"), tail.find("no")
    yes_index = yes_index if yes_index >= 0 else 10000
    no_index = no_index if no_index >= 0 else 10000
    if yes_index < no_index and yes_index < 5:
        return "yes"
    if no_index <= yes_index and no_index < 5:
        return "no"
    return None


def _clear_errors(utterance: dict, stage: str) -> None:
    remaining = [error for error in utterance.get("annotation_errors", []) if error.get("stage") != stage]
    if remaining:
        utterance["annotation_errors"] = remaining
    else:
        utterance.pop("annotation_errors", None)


def _error(utterance: dict, stage: str, code: str, message: str, result: EvaluationResult) -> None:
    utterance.setdefault("annotation_errors", []).append(
        {"stage": stage, "code": code, "message": message, "request_id": result.request_id,
         "request_fingerprint": result.details["_request_fingerprint"]}
    )
    recorder = result.details.get("_record_failure")
    if recorder is not None:
        recorder(result.details["_request_fingerprint"])


class EvaluationCheckers:
    """Run annotation stages without owning files, models, or cache policy."""

    def __init__(
        self,
        seed: int = 1234,
        batch_size: int = 64,
        num_prev_steps: int = 1,
        student_instruction: str = STUDENT_INSTRUCTION,
        verifier: Callable | None = None,
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if type(num_prev_steps) is not int or num_prev_steps < 1:
            raise ValueError("num_prev_steps must be a positive integer")
        self.seed = seed
        self.batch_size = batch_size
        self.num_prev_steps = num_prev_steps
        self.student_instruction = student_instruction
        self._verifier = verifier

    def _verify(self, output: str, answer: str) -> bool:
        if self._verifier is None:
            from .util_for_math_verify import MathVerifier

            self._verifier = MathVerifier()
        result = self._verifier(output, answer)
        if type(result) is not bool:
            raise TypeError("Mathematical verifier must return a bool")
        return result

    def _run_batches(
        self,
        items: Iterable[tuple[int, int, EvaluationRequest]],
        backend: Any,
    ) -> Iterator[tuple[int, int, EvaluationResult]]:
        """Bound prompt memory and match unordered results by their stable IDs."""
        iterator = iter(items)
        while batch := list(islice(iterator, self.batch_size)):
            requests = [item[2] for item in batch]
            expected = {request.request_id for request in requests}
            if len(expected) != len(requests):
                raise RuntimeError("Duplicate evaluation request IDs")
            # Infrastructure errors must abort the stage so it can be resumed.
            results = backend.generate(requests)
            by_id = {}
            for result in results:
                request_id = getattr(result, "request_id", None)
                if request_id not in expected or request_id in by_id:
                    raise RuntimeError("Evaluation backend returned unexpected or duplicate result IDs")
                by_id[request_id] = result
            if set(by_id) != expected:
                raise RuntimeError("Evaluation backend did not return every requested result ID")
            for q_id, index, request in batch:
                result = by_id[request.request_id]
                yield q_id, index, replace(result, details={
                    **result.details, "_request_fingerprint": fingerprint(asdict(request)),
                    "_record_failure": getattr(backend, "mark_failed", None),
                })

    @staticmethod
    def _result_failed(utterance: dict, stage: str, result: EvaluationResult) -> bool:
        if result.error is not None or result.error_code is not None:
            _error(utterance, stage, result.error_code or "backend_error", result.error or "Evaluation request failed", result)
            return True
        return False

    def annotate_overinformative(
        self, dialogues: dict[int, list[dict]], problems: dict[int, dict], backend: Any
    ) -> dict[int, list[dict]]:
        data = copy.deepcopy(dialogues)
        for q_id, dialogue in data.items():
            answer = problems[q_id]["answer"]
            for index, utterance in enumerate(dialogue):
                _clear_errors(utterance, "overinformative")
                if utterance["role"] == "teacher":
                    utterance["overinformative"] = False
                    utterance["answer_reachability_gain"] = 0 if index == 0 else None
                    utterance["teacher_judgment"] = None
                elif utterance["role"] == "student":
                    if "is_correct" not in utterance:
                        try:
                            utterance["is_correct"] = self._verify(utterance["content"], answer)
                        except Exception as exc:
                            raise RuntimeError(f"Cannot grade student answer for problem {q_id}, turn {index}") from exc
                    if type(utterance["is_correct"]) is not bool:
                        raise ValueError(f"Student is_correct must be a bool for problem {q_id}, turn {index}")

        def requests() -> Iterator[tuple[int, int, EvaluationRequest]]:
            for q_id, dialogue in data.items():
                for index, utterance in enumerate(dialogue):
                    if utterance["role"] != "student" or not utterance["is_correct"]:
                        continue
                    hints = []
                    for previous in range(index - 1, index - self.num_prev_steps * 2, -2):
                        if previous <= 0:
                            break
                        if dialogue[previous]["role"] != "teacher":
                            raise ValueError("A student answer must follow a teacher utterance")
                        hints.append(dialogue[previous]["content"].strip())
                    if not hints:
                        continue
                    # A requested probe is unknown until generation and grading succeed.
                    dialogue[index - 1]["overinformative"] = None
                    hint = "\n".join(reversed(hints))
                    yield q_id, index - 1, EvaluationRequest(
                        request_id=f"overinformative:{q_id}:{index - 1}",
                        messages=[
                            {"role": "user", "content": f"Problem: {problems[q_id]['question']}\n\nHint: {hint}"},
                            {"role": "assistant", "content": "So, the answer is"},
                        ],
                        system_prompt=OVERINFORMATIVE_INSTRUCTION,
                        seed=self.seed,
                    )

        for q_id, index, result in self._run_batches(requests(), backend):
            utterance = data[q_id][index]
            if self._result_failed(utterance, "overinformative", result):
                continue
            if not isinstance(result.text, str) or not result.text.strip():
                _error(utterance, "overinformative", "empty_response", "Hint-only completion is empty", result)
                continue
            try:
                # Preserve the original first-line-only grading policy.
                utterance["overinformative"] = self._verify(result.text.lstrip().split("\n")[0].strip(), problems[q_id]["answer"])
            except Exception:
                _error(utterance, "overinformative", "grading_error", "Hint-only answer could not be graded", result)
        return data

    def annotate_reachability(
        self, dialogues: dict[int, list[dict]], problems: dict[int, dict], backend: Any
    ) -> dict[int, list[dict]]:
        data = copy.deepcopy(dialogues)
        for dialogue in data.values():
            for utterance in dialogue:
                _clear_errors(utterance, "reachability")
                utterance.pop("reachability_details", None)
                if utterance["role"] in {"teacher", "student"}:
                    utterance["answer_reachability"] = None

        def requests() -> Iterator[tuple[int, int, EvaluationRequest]]:
            for q_id, dialogue in data.items():
                messages = []
                for index, utterance in enumerate(dialogue):
                    if utterance["role"] not in {"teacher", "student"}:
                        continue
                    messages.append({"role": "user" if utterance["role"] == "teacher" else "assistant", "content": utterance["content"]})
                    yield q_id, index, EvaluationRequest(
                        request_id=f"reachability:{q_id}:{index}",
                        messages=copy.deepcopy(messages),
                        system_prompt=self.student_instruction,
                        seed=self.seed,
                        kind="reachability",
                        answer=problems[q_id]["answer"],
                    )

        for q_id, index, result in self._run_batches(requests(), backend):
            dialogue, score = data[q_id], result.score
            utterance = dialogue[index]
            failed = self._result_failed(utterance, "reachability", result)
            if result.details:
                utterance["reachability_details"] = {
                    key: copy.deepcopy(value) for key, value in result.details.items() if not key.startswith("_")
                }
                skipped = result.details.get("skipped_token_count", 0)
                if isinstance(skipped, int) and skipped > 0:
                    _error(utterance, "reachability", "partial_logprobs", f"{skipped} answer-token log probabilities were unavailable", result)
            if not failed and (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)):
                _error(utterance, "reachability", "missing_score", "No finite answer reachability score was returned", result)
                failed = True
            if failed:
                # Preserve v3: reuse the same role's previous score, including
                # a score that itself came from this fallback.
                score = dialogue[index - 2].get("answer_reachability") if index >= 2 else None
            utterance["answer_reachability"] = score

        for dialogue in data.values():
            for index, utterance in enumerate(dialogue):
                if utterance["role"] != "teacher":
                    continue
                if index == 0:
                    utterance["answer_reachability_gain"] = 0
                    continue
                if dialogue[index - 1]["role"] != "student":
                    raise ValueError("A teacher feedback turn must follow a student utterance")
                previous = dialogue[index - 1]["answer_reachability"]
                current = utterance["answer_reachability"]
                utterance["answer_reachability_gain"] = None if previous is None or current is None else current - previous
        return data

    def annotate_teacher_judgment(
        self, dialogues: dict[int, list[dict]], backend: Any
    ) -> dict[int, list[dict]]:
        data = copy.deepcopy(dialogues)
        for dialogue in data.values():
            for utterance in dialogue:
                _clear_errors(utterance, "teacher_judgment")
                if utterance["role"] == "teacher":
                    utterance["teacher_judgment"] = None

        def requests() -> Iterator[tuple[int, int, EvaluationRequest]]:
            for q_id, dialogue in data.items():
                text = ""
                for index, utterance in enumerate(dialogue):
                    if utterance["role"] == "system":
                        continue
                    text += f"[{utterance['role'].upper()}]\n{utterance['content']}\n\n"
                    if utterance["role"] != "teacher" or index < 2:
                        continue
                    if dialogue[index - 1]["role"] != "student":
                        raise ValueError("A teacher feedback turn must follow a student utterance")
                    yield q_id, index, EvaluationRequest(
                        request_id=f"teacher_judgment:{q_id}:{index}",
                        messages=[{"role": "user", "content": text}, {"role": "assistant", "content": '{"label":'}],
                        system_prompt=TEACHER_JUDGMENT_CLASSIFICATION_INSTRUCTION,
                        seed=self.seed,
                    )

        for q_id, index, result in self._run_batches(requests(), backend):
            utterance = data[q_id][index]
            if self._result_failed(utterance, "teacher_judgment", result):
                continue
            label = parse_label(result.text)
            if label is None:
                _error(utterance, "teacher_judgment", "invalid_label", "Teacher judgment response did not contain a YES/NO label", result)
            else:
                utterance["teacher_judgment"] = label == "yes"
        return data
