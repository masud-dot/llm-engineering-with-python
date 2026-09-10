"""Turning a user's reaction into an evaluation case."""

from dataclasses import dataclass, field
from enum import Enum

from llmapp.eval.dataset import Case


class Verdict(str, Enum):
    GOOD = "good"
    BAD = "bad"
    WRONG_FACT = "wrong_fact"
    MISSING = "missing"
    UNSAFE = "unsafe"


@dataclass(frozen=True)
class Feedback:
    """What the user said, and what the request was."""

    request_id: str
    question: str
    verdict: Verdict
    principal: str = ""
    comment: str = ""
    retrieved_ids: tuple[str, ...] = ()
    prompt_id: str = ""


@dataclass
class FeedbackStore:
    """Collect reactions and promote the useful ones."""

    items: list[Feedback] = field(default_factory=list)

    def add(self, feedback: Feedback) -> None:
        self.items.append(feedback)

    @property
    def negative(self) -> list[Feedback]:
        return [
            f for f in self.items if f.verdict is not Verdict.GOOD
        ]

    def satisfaction(self) -> float:
        if not self.items:
            return 0.0
        good = sum(
            1 for f in self.items if f.verdict is Verdict.GOOD
        )
        return good / len(self.items)

    def to_cases(self) -> list[Case]:
        """Chapter 19's loop: a failure becomes a case."""
        cases: list[Case] = []
        for item in self.negative:
            cases.append(
                Case(
                    id=f"prod-{item.request_id}",
                    question=item.question,
                    relevant_ids=item.retrieved_ids,
                    answerable=item.verdict is not Verdict.MISSING,
                    tags=("production", item.verdict.value),
                    source="production",
                )
            )
        return cases
