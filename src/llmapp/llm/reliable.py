"""One wrapper that makes any LLMClient survive a bad day."""

import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from llmapp.llm.base import Completion, LLMClient, Message
from llmapp.llm.budget import SpendGuard
from llmapp.llm.errors import (
    CircuitOpen,
    PermanentError,
    TransientError,
)
from llmapp.llm.pricing import Rates, cost_usd, estimate_usd
from llmapp.obs.log import get_logger, redact

log = get_logger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    """How hard to try, and how long to wait between tries."""

    attempts: int = 3
    initial_delay_s: float = 0.5
    max_delay_s: float = 8.0
    jitter: float = 0.25


def backoff_delay(
    attempt: int,
    policy: RetryPolicy,
    rand: Callable[[], float] = random.random,
) -> float:
    """Exponential delay with jitter. attempt starts at 0."""
    raw = policy.initial_delay_s * (2.0**attempt)
    capped = min(raw, policy.max_delay_s)
    return capped * (1.0 - policy.jitter * rand())


@dataclass
class CircuitBreaker:
    """Stop calling a dependency that is clearly down."""

    failure_threshold: int = 5
    reset_after_s: float = 30.0
    failures: int = 0
    opened_at: float | None = None

    def allow(self, now: float) -> bool:
        if self.opened_at is None:
            return True
        if now - self.opened_at >= self.reset_after_s:
            self.opened_at = None
            self.failures = 0
            return True  # half-open: let one call through
        return False

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self, now: float) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = now


@dataclass
class ReliableClient:
    """Wrap a client with retries, a breaker, and accounting."""

    primary: LLMClient
    rates: Rates
    fallback: LLMClient | None = None
    policy: RetryPolicy = field(default_factory=RetryPolicy)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    guard: SpendGuard | None = None
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    rand: Callable[[], float] = random.random

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_output_tokens: int = 512,
    ) -> Completion:
        if self.guard is not None:
            approx_input = sum(
                len(m.content) for m in messages
            ) // 4
            self.guard.check(
                estimate_usd(
                    approx_input, max_output_tokens, self.rates
                )
            )
        try:
            return self._attempt_all(
                self.primary, messages, max_output_tokens, "primary"
            )
        except (TransientError, CircuitOpen):
            if self.fallback is None:
                raise
            log.warning("llm.fallback", reason="primary_exhausted")
            return self._attempt_all(
                self.fallback, messages, max_output_tokens,
                "fallback",
            )

    def _attempt_all(
        self,
        client: LLMClient,
        messages: Sequence[Message],
        max_output_tokens: int,
        tier: str,
    ) -> Completion:
        last: Exception | None = None
        for attempt in range(self.policy.attempts):
            now = self.clock()
            if not self.breaker.allow(now):
                raise CircuitOpen("breaker open; call not attempted")
            started = self.clock()
            try:
                result = client.complete(
                    messages, max_output_tokens=max_output_tokens
                )
            except PermanentError:
                self.breaker.record_success()  # server is healthy
                raise
            except TransientError as exc:
                last = exc
                self.breaker.record_failure(self.clock())
                log.warning(
                    "llm.retry",
                    tier=tier,
                    attempt=attempt + 1,
                    error=redact(str(exc)),
                )
                if attempt == self.policy.attempts - 1:
                    break
                wait = exc.retry_after_s
                if wait is None:
                    wait = backoff_delay(
                        attempt, self.policy, self.rand
                    )
                self.sleep(wait)
                continue
            self.breaker.record_success()
            self._record(result, tier, self.clock() - started)
            return result
        assert last is not None
        raise last

    def _record(
        self, result: Completion, tier: str, elapsed_s: float
    ) -> None:
        spent = cost_usd(result, self.rates)
        if self.guard is not None:
            self.guard.record(spent)
        log.info(
            "llm.call",
            tier=tier,
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=round(spent, 6),
            elapsed_s=round(elapsed_s, 3),
            finish_reason=result.finish_reason,
        )
