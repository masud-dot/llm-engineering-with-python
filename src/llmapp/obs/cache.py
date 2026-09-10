"""Caching model responses, without leaking between users."""

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from llmapp.llm.base import Completion, Message


class CacheKeyError(ValueError):
    """A cache key was built without everything it needs."""


def cache_key(
    *,
    principal: str,
    model: str,
    prompt_id: str,
    messages: Sequence[Message],
    scope: Sequence[str] = (),
) -> str:
    """Identify a cached answer.

    The principal is part of the key, not an afterthought.
    A cache keyed on the question alone serves one user's
    answer to another and quietly undoes the permission
    filtering built in Chapter 14.
    """
    if not principal:
        raise CacheKeyError(
            "a cache key needs a principal; see Chapter 21"
        )
    if not model:
        raise CacheKeyError("a cache key needs the model")
    body = "\x00".join(
        f"{m.role}:{m.content}" for m in messages
    )
    parts = [principal, model, prompt_id, body, *scope]
    blob = "\x1f".join(parts).encode()
    return hashlib.sha256(blob).hexdigest()[:32]


@dataclass
class Entry:
    """One cached response and when it was stored."""

    completion: Completion
    stored_at: float


@dataclass
class ExactCache:
    """Same principal, same prompt, same model, same answer."""

    ttl_seconds: float = 3600.0
    max_entries: int = 10_000
    clock: Callable[[], float] = time.monotonic
    entries: dict[str, Entry] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    expired: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def get(self, key: str) -> Completion | None:
        entry = self.entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        if self.clock() - entry.stored_at > self.ttl_seconds:
            del self.entries[key]
            self.expired += 1
            self.misses += 1
            return None
        self.hits += 1
        return entry.completion

    def put(self, key: str, completion: Completion) -> None:
        if len(self.entries) >= self.max_entries:
            oldest = min(
                self.entries,
                key=lambda k: self.entries[k].stored_at,
            )
            del self.entries[oldest]
        self.entries[key] = Entry(completion, self.clock())

    def invalidate(self, prefix: str = "") -> int:
        """Drop everything, or every key with a prefix."""
        doomed = [
            key for key in self.entries if key.startswith(prefix)
        ]
        for key in doomed:
            del self.entries[key]
        return len(doomed)


@dataclass(frozen=True)
class CacheStats:
    """What a cache is worth, in one line."""

    hits: int
    misses: int
    expired: int
    saved_usd: float

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def __str__(self) -> str:
        return (
            f"hit rate {self.hit_rate:.0%} "
            f"({self.hits}/{self.hits + self.misses}), "
            f"saved ${self.saved_usd:.4f}"
        )
