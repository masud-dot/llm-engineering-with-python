"""Using a model to score output, with the bias controls."""

import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from llmapp.llm.base import LLMClient
from llmapp.prompts.registry import PromptRegistry
from llmapp.schemas.validate import parse_or_repair

DEFAULT_RUBRIC = (
    "5 = fully supported by the evidence and answers the "
    "question; 3 = partly supported or partly answers it; "
    "1 = unsupported, wrong, or does not answer it."
)


class Verdict(BaseModel):
    """A single-answer score."""

    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=1, le=5)
    reason: str = Field(min_length=1, max_length=300)


class Choice(BaseModel):
    """A pairwise comparison."""

    model_config = ConfigDict(extra="forbid")

    winner: Literal["A", "B", "tie"]
    reason: str = Field(min_length=1, max_length=300)


@dataclass
class Judge:
    """Scores answers. Never answers the question."""

    client: LLMClient
    registry: PromptRegistry
    rubric: str = DEFAULT_RUBRIC

    def score(
        self, question: str, answer: str, evidence: str
    ) -> Verdict:
        prompt = self.registry.get("eval_judge", 1).render(
            question=question,
            answer=answer,
            evidence=evidence,
            rubric=self.rubric,
        )
        verdict, _ = parse_or_repair(
            self.client, prompt.messages, Verdict
        )
        return verdict

    def compare(
        self, question: str, left: str, right: str
    ) -> Choice:
        """One ordering. Prefer compare_both_ways."""
        prompt = self.registry.get("eval_pairwise", 1).render(
            question=question,
            left=left,
            right=right,
            rubric=self.rubric,
        )
        choice, _ = parse_or_repair(
            self.client, prompt.messages, Choice
        )
        return choice

    def compare_both_ways(
        self, question: str, left: str, right: str
    ) -> tuple[str, bool]:
        """Judge twice with the order swapped.

        Returns the winner and whether the two runs agreed.
        A judge with position bias disagrees with itself, and
        the disagreement is the signal.
        """
        first = self.compare(question, left, right)
        second = self.compare(question, right, left)
        flipped = {"A": "B", "B": "A", "tie": "tie"}
        second_winner = flipped[second.winner]
        if first.winner == second_winner:
            return first.winner, True
        return "tie", False


def position_bias(pairs: list[tuple[str, bool]]) -> float:
    """Share of comparisons where the judge contradicted itself.

    Zero means the order made no difference. One means the
    judge chose by position every time.
    """
    if not pairs:
        raise ValueError("no comparisons to measure")
    disagreed = sum(1 for _, agreed in pairs if not agreed)
    return disagreed / len(pairs)


def win_rate(results: list[str], candidate: str = "B") -> float:
    """Share of decided comparisons the candidate won."""
    decided = [r for r in results if r != "tie"]
    if not decided:
        return 0.0
    return sum(1 for r in decided if r == candidate) / len(decided)
