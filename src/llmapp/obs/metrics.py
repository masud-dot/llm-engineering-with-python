"""The metric set, and the alerts that read it."""

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class Series:
    """One metric's observations, with percentiles."""

    name: str
    values: list[float] = field(default_factory=list)

    def observe(self, value: float) -> None:
        self.values.append(value)

    def percentile(self, fraction: float) -> float:
        if not self.values:
            return 0.0
        ordered = sorted(self.values)
        index = min(
            len(ordered) - 1,
            int(round(fraction * (len(ordered) - 1))),
        )
        return ordered[index]

    @property
    def mean(self) -> float:
        return statistics.fmean(self.values) if self.values else 0.0

    @property
    def total(self) -> float:
        return sum(self.values)


@dataclass
class Metrics:
    """Everything worth watching, in one place."""

    latency_ms: Series = field(
        default_factory=lambda: Series("latency_ms")
    )
    ttft_ms: Series = field(
        default_factory=lambda: Series("ttft_ms")
    )
    cost_usd: Series = field(
        default_factory=lambda: Series("cost_usd")
    )
    input_tokens: Series = field(
        default_factory=lambda: Series("input_tokens")
    )
    output_tokens: Series = field(
        default_factory=lambda: Series("output_tokens")
    )
    counters: dict[str, int] = field(default_factory=dict)

    def count(self, name: str, amount: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + amount

    def rate(self, numerator: str, denominator: str) -> float:
        total = self.counters.get(denominator, 0)
        if not total:
            return 0.0
        return self.counters.get(numerator, 0) / total

    def summary(self) -> str:
        return (
            f"requests {self.counters.get('requests', 0)}  "
            f"p50 {self.latency_ms.percentile(0.5):.0f}ms  "
            f"p95 {self.latency_ms.percentile(0.95):.0f}ms  "
            f"cost/req ${self.cost_usd.mean:.4f}  "
            f"errors {self.rate('errors', 'requests'):.0%}  "
            f"abstain {self.rate('abstained', 'requests'):.0%}"
        )


class Severity(str, Enum):
    OK = "ok"
    WARN = "warn"
    PAGE = "page"


@dataclass(frozen=True)
class Rule:
    """One alert: a metric, a bound, and what it means."""

    name: str
    value: float
    ceiling: float | None = None
    floor: float | None = None
    severity: Severity = Severity.WARN

    def fires(self) -> bool:
        if self.ceiling is not None and self.value > self.ceiling:
            return True
        return self.floor is not None and self.value < self.floor

    def describe(self) -> str:
        bound = (
            f"above {self.ceiling}"
            if self.ceiling is not None
            and self.value > self.ceiling
            else f"below {self.floor}"
        )
        return f"{self.name} {self.value:.3g} is {bound}"


def evaluate(rules: Sequence[Rule]) -> list[Rule]:
    """Every rule that is firing, worst first."""
    order = {
        Severity.PAGE: 0,
        Severity.WARN: 1,
        Severity.OK: 2,
    }
    firing = [rule for rule in rules if rule.fires()]
    return sorted(firing, key=lambda r: order[r.severity])


def change_rate(previous: float, current: float) -> float:
    """Relative movement, for alerting on the derivative.

    A metric five percent below a threshold somebody set six
    months ago is less informative than one that moved forty
    percent since yesterday.
    """
    if previous == 0:
        return 0.0 if current == 0 else 1.0
    return (current - previous) / previous
