"""Rate limiting at the edge, per principal."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from fastapi import HTTPException, status


class RateLimited(HTTPException):
    """429 with a Retry-After the caller can honor."""

    def __init__(self, retry_after: int) -> None:
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )


@dataclass
class SlidingWindow:
    """N requests per window, per key, in this process.

    A deployment with several workers needs a shared store;
    the interface is kept small so it can be replaced.
    """

    limit: int
    window_seconds: float = 60.0
    clock: Callable[[], float] = time.monotonic
    seen: dict[str, list[float]] = field(default_factory=dict)

    def check(self, key: str) -> None:
        now = self.clock()
        cutoff = now - self.window_seconds
        recent = [t for t in self.seen.get(key, []) if t > cutoff]
        if len(recent) >= self.limit:
            oldest = min(recent)
            wait = int(self.window_seconds - (now - oldest)) + 1
            self.seen[key] = recent
            raise RateLimited(retry_after=wait)
        recent.append(now)
        self.seen[key] = recent

    def remaining(self, key: str) -> int:
        cutoff = self.clock() - self.window_seconds
        recent = [t for t in self.seen.get(key, []) if t > cutoff]
        return max(0, self.limit - len(recent))
