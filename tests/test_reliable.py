import pytest

from llmapp.llm.base import Completion, Message
from llmapp.llm.errors import PermanentError, TransientError
from llmapp.llm.pricing import Rates
from llmapp.llm.reliable import (
    CircuitBreaker,
    ReliableClient,
    RetryPolicy,
    backoff_delay,
)

ASK = [Message(role="user", content="hello")]
RATES = Rates(input_per_mtok=3.0, output_per_mtok=15.0)


class FlakyClient:
    """Fail a fixed number of times, then succeed."""

    def __init__(
        self, failures: int, error: Exception | None = None
    ) -> None:
        self.remaining = failures
        self.error = error or TransientError("503 upstream")
        self.calls = 0

    def complete(
        self,
        messages: object,
        *,
        max_output_tokens: int = 512,
    ) -> Completion:
        self.calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise self.error
        return Completion("ok", 100, 20, "flaky", "stop")


def build(primary: object, **kw: object) -> ReliableClient:
    return ReliableClient(
        primary=primary,  # type: ignore[arg-type]
        rates=RATES,
        sleep=lambda _s: None,  # never sleep in tests
        rand=lambda: 0.0,
        **kw,  # type: ignore[arg-type]
    )


def test_backoff_grows_and_is_capped() -> None:
    policy = RetryPolicy(initial_delay_s=0.5, max_delay_s=8.0)
    fixed = [backoff_delay(n, policy, lambda: 0.0) for n in range(6)]
    assert fixed == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0]


def test_jitter_only_reduces_the_delay() -> None:
    policy = RetryPolicy(initial_delay_s=1.0, jitter=0.25)
    full = backoff_delay(0, policy, lambda: 1.0)
    none = backoff_delay(0, policy, lambda: 0.0)
    assert full == pytest.approx(0.75)
    assert none == pytest.approx(1.0)


def test_transient_failure_is_retried_then_succeeds() -> None:
    flaky = FlakyClient(failures=2)
    result = build(flaky).complete(ASK)
    assert result.text == "ok"
    assert flaky.calls == 3


def test_permanent_failure_is_not_retried() -> None:
    bad = FlakyClient(failures=1, error=PermanentError("400"))
    with pytest.raises(PermanentError):
        build(bad).complete(ASK)
    assert bad.calls == 1


def test_attempts_are_bounded() -> None:
    always = FlakyClient(failures=99)
    with pytest.raises(TransientError):
        build(always, policy=RetryPolicy(attempts=3)).complete(ASK)
    assert always.calls == 3


def test_fallback_runs_when_primary_is_exhausted() -> None:
    primary = FlakyClient(failures=99)
    secondary = FlakyClient(failures=0)
    client = build(
        primary,
        fallback=secondary,
        policy=RetryPolicy(attempts=2),
    )
    assert client.complete(ASK).text == "ok"
    assert primary.calls == 2
    assert secondary.calls == 1


def test_breaker_opens_and_stops_calling() -> None:
    always = FlakyClient(failures=99)
    breaker = CircuitBreaker(failure_threshold=2, reset_after_s=30)
    client = build(
        always,
        policy=RetryPolicy(attempts=5),
        breaker=breaker,
        clock=lambda: 100.0,
    )
    with pytest.raises(TransientError):
        client.complete(ASK)
    assert always.calls == 2
    assert breaker.opened_at == 100.0


def test_breaker_half_opens_after_the_reset_window() -> None:
    breaker = CircuitBreaker(failure_threshold=1, reset_after_s=30)
    breaker.record_failure(now=100.0)
    assert breaker.allow(now=120.0) is False
    assert breaker.allow(now=131.0) is True
