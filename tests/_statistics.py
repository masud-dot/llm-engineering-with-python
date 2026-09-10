"""Assertions for output that is right most of the time."""
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Trials:
    """The outcome of running a check N times."""

    passed: int
    total: int

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


def repeat(check: Callable[[], bool], times: int = 20) -> Trials:
    """Run a probabilistic check repeatedly."""
    if times < 1:
        raise ValueError("times must be at least 1")
    passed = sum(1 for _ in range(times) if check())
    return Trials(passed=passed, total=times)


def assert_pass_rate(
    check: Callable[[], bool],
    *,
    times: int = 20,
    minimum: float = 0.9,
) -> Trials:
    """Fail if the observed rate is below the floor.

    The floor is a decision about acceptable quality, not a
    statistical guarantee: with 20 trials a true rate of 0.95
    still fails this check some of the time.
    """
    trials = repeat(check, times)
    if trials.rate < minimum:
        raise AssertionError(
            f"passed {trials.passed}/{trials.total} "
            f"({trials.rate:.0%}), floor is {minimum:.0%}"
        )
    return trials


def within(value: float, expected: float, band: float) -> bool:
    """True when a measurement sits inside a tolerance band."""
    return abs(value - expected) <= band
