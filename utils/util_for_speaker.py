"""Role conversion independent of whether a model is local or remote."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .util_for_types import GenerationRequest

DIALOGUE_FINISH_SYMBOL = "### DIALOGUE FINISHED ###"
DIALOGUE_UNFINISH_SYMBOL = "### DIALOGUE NOT FINISHED ###"


@dataclass(frozen=True)
class Speaker:
    role: str
    instruction: str
    seed: int

    def request(self, q_id: int, message: list[dict]) -> GenerationRequest:
        if self.role not in {"teacher", "student"}:
            raise ValueError(f"Invalid speaker role: {self.role}")
        converted = []
        for utterance in message:
            role = utterance["role"]
            if role not in {"teacher", "student"}:
                raise ValueError("Cannot generate from a finished or invalid dialogue")
            converted.append(
                {
                    "role": "assistant" if role == self.role else "user",
                    "content": utterance["content"],
                }
            )
        turn = len(message)
        # Stable across Python processes, subsets, batch sizes and teacher comparisons.
        seed_material = f"{self.seed}:{self.role}:{q_id}:{turn}".encode()
        request_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big")
        return GenerationRequest(
            request_id=f"q{q_id}_t{turn}_{self.role}",
            messages=converted,
            system_prompt=self.instruction,
            seed=request_seed,
        )
