"""Send the cheap model first, escalate when it is not enough."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from llmapp.llm.base import Completion, LLMClient, Message
from llmapp.llm.pricing import Rates, cost_usd

Escalate = Callable[[Completion], bool]


@dataclass(frozen=True)
class Tier:
    """One model in the ladder, with its price."""

    name: str
    client: LLMClient
    rates: Rates


@dataclass
class RoutingResult:
    """What answered, and what it cost to get there."""

    completion: Completion
    tier: str
    escalated: bool
    cost_usd: float
    attempts: tuple[str, ...]


@dataclass
class ModelRouter:
    """Try each tier in order until one is good enough."""

    tiers: Sequence[Tier]
    should_escalate: Escalate
    calls: list[RoutingResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.tiers:
            raise ValueError("a router needs at least one tier")

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_output_tokens: int = 512,
    ) -> RoutingResult:
        attempts: list[str] = []
        spent = 0.0
        result: RoutingResult | None = None
        for index, tier in enumerate(self.tiers):
            attempts.append(tier.name)
            completion = tier.client.complete(
                messages, max_output_tokens=max_output_tokens
            )
            spent += cost_usd(completion, tier.rates)
            last = index == len(self.tiers) - 1
            if last or not self.should_escalate(completion):
                result = RoutingResult(
                    completion=completion,
                    tier=tier.name,
                    escalated=index > 0,
                    cost_usd=spent,
                    attempts=tuple(attempts),
                )
                break
        assert result is not None
        self.calls.append(result)
        return result

    @property
    def escalation_rate(self) -> float:
        if not self.calls:
            return 0.0
        return sum(
            1 for c in self.calls if c.escalated
        ) / len(self.calls)

    @property
    def total_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)


def escalate_on_refusal(marker: str = "i don't know") -> Escalate:
    """A cheap, deterministic escalation signal."""

    def check(completion: Completion) -> bool:
        return marker in completion.text.lower()

    return check


def escalate_on_short_answer(minimum: int = 3) -> Escalate:
    """Too few tokens usually means the model gave up."""

    def check(completion: Completion) -> bool:
        return completion.output_tokens < minimum

    return check


def any_of(*checks: Escalate) -> Escalate:
    def check(completion: Completion) -> bool:
        return any(one(completion) for one in checks)

    return check
