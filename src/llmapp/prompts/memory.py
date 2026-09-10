"""Four ways to remember a conversation, with four costs."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from llmapp.llm.base import LLMClient, Message

SUMMARY_INSTRUCTION = (
    "Summarize the conversation so far in under 80 words. "
    "Keep every identifier, number, name, and commitment "
    "exactly as written. Drop pleasantries."
)


class Memory(Protocol):
    """What a conversation strategy must provide."""

    def add(self, message: Message) -> None: ...

    def recall(self) -> list[Message]: ...


@dataclass
class BufferMemory:
    """Keep everything. Simple, and unbounded."""

    history: list[Message] = field(default_factory=list)

    def add(self, message: Message) -> None:
        self.history.append(message)

    def recall(self) -> list[Message]:
        return list(self.history)


@dataclass
class WindowMemory:
    """Keep the last N turns. Bounded, and forgetful."""

    max_turns: int = 6
    history: list[Message] = field(default_factory=list)

    def add(self, message: Message) -> None:
        self.history.append(message)

    def recall(self) -> list[Message]:
        return self.history[-(self.max_turns * 2) :]


@dataclass
class SummaryMemory:
    """Fold old turns into a rolling summary."""

    client: LLMClient
    keep_turns: int = 3
    summarize_after: int = 6
    summary: str = ""
    history: list[Message] = field(default_factory=list)

    def add(self, message: Message) -> None:
        self.history.append(message)
        if len(self.history) > self.summarize_after * 2:
            self._compact()

    def _compact(self) -> None:
        keep = self.keep_turns * 2
        old, recent = self.history[:-keep], self.history[-keep:]
        transcript = "\n".join(
            f"{m.role}: {m.content}" for m in old
        )
        if self.summary:
            transcript = (
                f"Previous summary: {self.summary}\n{transcript}"
            )
        result = self.client.complete(
            [
                Message(role="system", content=SUMMARY_INSTRUCTION),
                Message(role="user", content=transcript),
            ]
        )
        self.summary = result.text.strip()
        self.history = recent

    def recall(self) -> list[Message]:
        head: list[Message] = []
        if self.summary:
            head.append(
                Message(
                    role="system",
                    content=f"Conversation so far: {self.summary}",
                )
            )
        return head + list(self.history)


@dataclass
class HybridMemory:
    """Pinned facts, a rolling summary, and recent turns."""

    inner: SummaryMemory
    facts: dict[str, str] = field(default_factory=dict)

    def pin(self, key: str, value: str) -> None:
        """Facts here are never summarized away."""
        self.facts[key] = value

    def add(self, message: Message) -> None:
        self.inner.add(message)

    def recall(self) -> list[Message]:
        head: list[Message] = []
        if self.facts:
            lines = "\n".join(
                f"{k}: {v}" for k, v in self.facts.items()
            )
            head.append(
                Message(
                    role="system",
                    content=f"Established facts:\n{lines}",
                )
            )
        return head + self.inner.recall()


def transcript_tokens(messages: Sequence[Message]) -> int:
    """Token cost of a recalled payload."""
    from llmapp.prompts.context import count_tokens

    return sum(count_tokens(m.content) for m in messages)
