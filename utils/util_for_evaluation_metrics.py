"""Pure aggregation of the original tutoring evaluation policies.

ECR/FEI use the first correct student answer. ARG uses only initially incorrect
dialogues that eventually succeed, compares consecutive *teacher* reachability
scores, includes the closing teacher, freezes after overinformative feedback,
and carries forward missing teacher scores. Absent later turns contribute zero
to per-turn means. Slot zero contains the initial reachability, as in the
original report. Missing teacher judgments count as incorrect predictions.

An unknown OI decision makes its dependent aggregate metrics unavailable. An
unavailable initial reachability likewise withholds ARG rather than silently
dropping that dialogue. Empty denominators return zero with sample counts.
No input records are modified, and no files or model libraries are accessed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class _Dialogue:
    q_id: int
    turns: tuple[dict[str, Any], ...]
    first_correct_feedback: int | None
    student_count: int
    teacher_count: int


@dataclass
class _Diagnostics:
    total_dialogues: int = 0
    initial_correct_dialogues: int = 0
    unresolved_dialogues: int = 0
    missing_initial_student_dialogues: int = 0
    arg_eligible_dialogues: int = 0
    arg_valid_dialogues: int = 0
    arg_missing_initial_reachability_dialogues: int = 0
    arg_missing_teacher_reachability_count: int = 0
    arg_overinformative_dialogues: int = 0
    overinformative_unknown_corrected_dialogues: int = 0
    arg_unknown_overinformative_dialogues: int = 0
    arg_zero_padded_turn_count: int = 0
    teacher_judgment_total_count: int = 0
    teacher_judgment_valid_count: int = 0
    teacher_judgment_missing_count: int = 0
    configured_max_feedback_count: int = 0
    observed_max_teacher_ordinal: int = 0
    reported_max_feedback_count: int = 0


@dataclass
class _Confusion:
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    def add(self, truth: bool, prediction: bool) -> None:
        if truth:
            if prediction:
                self.true_positive += 1
            else:
                self.false_negative += 1
        elif prediction:
            self.false_positive += 1
        else:
            self.true_negative += 1

    def scores(self) -> dict[str, float]:
        tp, fp = self.true_positive, self.false_positive
        tn, fn = self.true_negative, self.false_negative
        return {
            "yes_precision": _divide(tp, tp + fp),
            "yes_recall": _divide(tp, tp + fn),
            "yes_f1_score": _divide(2 * tp, 2 * tp + fp + fn),
            "no_precision": _divide(tn, tn + fn),
            "no_recall": _divide(tn, tn + fp),
            "no_f1_score": _divide(2 * tn, 2 * tn + fp + fn),
            "accuracy": _divide(tp + tn, tp + fp + tn + fn),
        }


def _divide(numerator: float, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _score(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number or None")
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number or None")
    return float(value)


def _validate_dialogue(q_id: int, messages: list[dict[str, Any]]) -> _Dialogue:
    if type(q_id) is not int or q_id < 0:
        raise ValueError("Dialogue IDs must be non-negative integers")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Dialogue {q_id} must be a non-empty message list")
    turns: list[dict[str, Any]] = []
    first_correct: int | None = None
    student_count = teacher_count = 0
    for index, utterance in enumerate(messages):
        if not isinstance(utterance, dict):
            raise ValueError(f"Dialogue {q_id}, turn {index} must be a dictionary")
        role = utterance.get("role")
        if role == "system":
            if index != len(messages) - 1 or not turns:
                raise ValueError(f"Dialogue {q_id} permits a system marker only at the end")
            continue
        expected_role = "teacher" if index % 2 == 0 else "student"
        if role != expected_role:
            raise ValueError(f"Dialogue {q_id}, turn {index}: expected {expected_role}")
        if role == "student":
            if type(utterance.get("is_correct")) is not bool:
                raise ValueError(f"Dialogue {q_id}, turn {index}: is_correct must be boolean")
            if utterance["is_correct"] and first_correct is None:
                first_correct = index // 2
            student_count += 1
        else:
            oi = utterance.get("overinformative", False)
            if oi is not None and type(oi) is not bool:
                raise ValueError(f"Dialogue {q_id}, turn {index}: overinformative must be boolean or None")
            judgment = utterance.get("teacher_judgment")
            if judgment is not None and type(judgment) is not bool:
                raise ValueError(f"Dialogue {q_id}, turn {index}: teacher_judgment must be boolean or None")
            for field in ("answer_reachability", "answer_reachability_gain"):
                if field in utterance:
                    _score(utterance[field], f"Dialogue {q_id}, turn {index}: {field}")
            teacher_count += 1
        turns.append(utterance)
    return _Dialogue(q_id, tuple(turns), first_correct, student_count, teacher_count)


def evaluate_dialogues(
    dialogues: dict[int, list[dict[str, Any]]], max_feedback_count: int
) -> dict[str, Any]:
    """Return legacy flat metric fields and explicit validity diagnostics.

    The configured budget controls zero-padding, not truncation. Maps extend to
    the largest observed teacher ordinal, including a final teacher at budget+1.
    Students require boolean ``is_correct``. Teacher judgments may be None;
    teacher overinformative flags default to False; None means an unknown probe.
    ARG-eligible dialogues need
    both reachability fields on each teacher, with initial gain equal to zero.
    Failed initial student generation (teacher plus end marker) is supported.

    ``q_id2feedback_count`` is zero for an initially correct answer, positive for
    an ordinary correction, -1 for an overinformative correction, and None when
    unresolved or its OI decision is unknown. Diagnostics distinguish unknown
    corrections, and OI-dependent aggregates remain unavailable for that group.
    """
    if type(max_feedback_count) is not int or max_feedback_count < 0:
        raise ValueError("max_feedback_count must be a non-negative integer")
    if not isinstance(dialogues, dict):
        raise ValueError("dialogues must map integer IDs to message lists")
    records = [_validate_dialogue(q_id, messages) for q_id, messages in dialogues.items()]
    observed_max = max((record.teacher_count - 1 for record in records), default=0)
    maximum = max(max_feedback_count, observed_max)
    corrected_counts = dict.fromkeys(range(maximum + 1), 0)
    overinformative_counts = corrected_counts.copy()
    unknown_counts = corrected_counts.copy()
    arg_sums = dict.fromkeys(range(maximum + 1), 0.0)
    q_id2feedback_count: dict[int, int | None] = {}
    diagnostics = _Diagnostics(
        total_dialogues=len(records),
        configured_max_feedback_count=max_feedback_count,
        observed_max_teacher_ordinal=observed_max,
        reported_max_feedback_count=maximum,
    )
    confusion = _Confusion()
    doc_arg_sum = 0.0

    for record in records:
        feedback_count = record.first_correct_feedback
        q_id2feedback_count[record.q_id] = None
        if not record.student_count:
            diagnostics.missing_initial_student_dialogues += 1
        if feedback_count is None:
            diagnostics.unresolved_dialogues += 1
        else:
            corrected_counts[feedback_count] += 1
            overinformative = record.turns[2 * feedback_count].get("overinformative", False)
            if overinformative:
                overinformative_counts[feedback_count] += 1
            elif overinformative is None and feedback_count:
                unknown_counts[feedback_count] += 1
                diagnostics.overinformative_unknown_corrected_dialogues += 1
            q_id2feedback_count[record.q_id] = (
                None if feedback_count and overinformative is None
                else -1 if feedback_count and overinformative else feedback_count
            )
            if feedback_count == 0:
                diagnostics.initial_correct_dialogues += 1

        # The original target is sticky: once a student is correct, later
        # teacher judgments are compared with True as well.
        student_correct = False
        for index, utterance in enumerate(record.turns):
            if utterance["role"] == "student":
                student_correct = student_correct or utterance["is_correct"]
            elif index:
                judgment = utterance.get("teacher_judgment")
                diagnostics.teacher_judgment_total_count += 1
                if judgment is None:
                    diagnostics.teacher_judgment_missing_count += 1
                    judgment = not student_correct
                else:
                    diagnostics.teacher_judgment_valid_count += 1
                confusion.add(student_correct, judgment)

        if feedback_count is None or feedback_count == 0:
            continue
        diagnostics.arg_eligible_dialogues += 1
        teachers = record.turns[::2]
        for ordinal, teacher in enumerate(teachers):
            if not {"answer_reachability", "answer_reachability_gain"} <= teacher.keys():
                raise ValueError(
                    f"Dialogue {record.q_id}, teacher {ordinal}: missing reachability fields"
                )
        if teachers[0]["answer_reachability_gain"] != 0:
            raise ValueError(f"Dialogue {record.q_id}: initial answer_reachability_gain must be zero")

        # OI rate uses the full eligible population even when an ARG baseline is
        # missing: the OI labels themselves remain observable.
        if any(teacher.get("overinformative", False) for teacher in teachers[1:]):
            diagnostics.arg_overinformative_dialogues += 1
        unknown_oi = any(teacher.get("overinformative", False) is None for teacher in teachers[1:])
        if unknown_oi:
            diagnostics.arg_unknown_overinformative_dialogues += 1
        diagnostics.arg_missing_teacher_reachability_count += sum(
            teacher["answer_reachability"] is None for teacher in teachers[1:]
        )
        initial = teachers[0]["answer_reachability"]
        if initial is None:
            diagnostics.arg_missing_initial_reachability_dialogues += 1
            continue
        if unknown_oi:
            continue

        diagnostics.arg_valid_dialogues += 1
        arg_sums[0] += initial
        previous = initial
        overinformative = False
        for ordinal, teacher in enumerate(teachers[1:], start=1):
            overinformative = overinformative or teacher.get("overinformative", False)
            current = teacher["answer_reachability"]
            if overinformative or current is None:
                current = previous
            arg_sums[ordinal] += current - previous
            previous = current
        doc_arg_sum += previous - initial
        # Zero-padding needs no additions; dividing each sum by all valid ARG
        # dialogues retains terminated dialogues in every later-turn mean.
        diagnostics.arg_zero_padded_turn_count += maximum + 1 - len(teachers)

    initial_incorrect = len(records) - corrected_counts[0]
    corrected = sum(count for ordinal, count in corrected_counts.items() if ordinal)
    overinformative_corrected = sum(
        count for ordinal, count in overinformative_counts.items() if ordinal
    )
    real_counts = {
        ordinal: None if unknown_counts[ordinal] else count - overinformative_counts[ordinal]
        for ordinal, count in corrected_counts.items()
    }
    inverse_sum = sum(count / ordinal for ordinal, count in corrected_counts.items() if ordinal)
    real_inverse_sum = sum(count / ordinal for ordinal, count in real_counts.items() if ordinal and count is not None)
    oi_available = not diagnostics.overinformative_unknown_corrected_dialogues
    arg_available = not (
        diagnostics.arg_unknown_overinformative_dialogues
        or diagnostics.arg_missing_initial_reachability_dialogues
    )
    result: dict[str, Any] = {
        "initial_incorrect_count": initial_incorrect,
        "corrected_count": corrected,
        "error_correction_rate": round(_divide(corrected, initial_incorrect), 4),
        "feedback_efficiency_index": round(_divide(inverse_sum, initial_incorrect), 4),
        "real_corrected_count": corrected - overinformative_corrected if oi_available else None,
        "real_error_correction_rate": round(
            _divide(corrected - overinformative_corrected, initial_incorrect), 4
        ) if oi_available else None,
        "real_feedback_efficiency_index": round(_divide(real_inverse_sum, initial_incorrect), 4) if oi_available else None,
        "feedback_count2corrected_count": corrected_counts,
        "feedback_count2overinformative_count": overinformative_counts,
        "feedback_count2unknown_overinformative_count": unknown_counts,
        "feedback_count2real_corrected_count": real_counts,
        "q_id2feedback_count": q_id2feedback_count,
        "overinformative_rate": round(
            _divide(diagnostics.arg_overinformative_dialogues, diagnostics.arg_eligible_dialogues), 4
        ) if not diagnostics.arg_unknown_overinformative_dialogues else None,
        "avg_doc_answer_reachability_gain": round(
            _divide(doc_arg_sum, diagnostics.arg_valid_dialogues), 4
        ) if arg_available else None,
        "feedback_count2avg_answer_reachability_gain": {
            ordinal: round(_divide(value, diagnostics.arg_valid_dialogues), 4) if arg_available else None
            for ordinal, value in arg_sums.items()
        },
        **confusion.scores(),
        "diagnostics": asdict(diagnostics),
    }
    return result


def winrate_normalize_one_student(
    teacher_ids: Sequence[str], values: Sequence[int | float], *, tie_value: float = 0.5
) -> dict[str, float]:
    """Compare teacher summary scores; ties score 0.5, a sole teacher gets 0.5.

    This is the original within-student normalization, not a per-problem win
    rate. Callers should preserve raw ARG separately instead of losing it.
    """
    if len(teacher_ids) != len(values):
        raise ValueError("Teacher IDs and values must have the same length")
    if any(not isinstance(teacher, str) or not teacher for teacher in teacher_ids):
        raise ValueError("Teacher IDs must be non-empty strings")
    if len(set(teacher_ids)) != len(teacher_ids):
        raise ValueError("Teacher IDs must be unique")
    scores: list[float] = []
    for value in values:
        score = _score(value, "Teacher score")
        if score is None:
            raise ValueError("Teacher scores cannot be None")
        scores.append(score)
    tie = _score(tie_value, "tie_value")
    if tie is None or not 0 <= tie <= 1:
        raise ValueError("tie_value must be between zero and one")
    if len(teacher_ids) < 2:
        return {teacher_ids[0]: 0.5} if teacher_ids else {}
    return {
        teacher: sum(
            1.0 if score > other else tie if score == other else 0.0
            for j, other in enumerate(scores)
            if i != j
        ) / (len(scores) - 1)
        for i, (teacher, score) in enumerate(zip(teacher_ids, scores))
    }
