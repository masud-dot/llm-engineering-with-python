"""Multi-turn state, bounded on purpose."""

from collections.abc import Sequence
from dataclasses import dataclass, field

from llmapp.llm.base import Completion, LLMClient, Message


@dataclass
class Conversation:
    """Hold history and send it back on every turn."""

    client: LLMClient
    system: str = ""
    max_turns: int = 10
    history: list[Message] = field(default_factory=list)

    def messages(self) -> list[Message]:
        """The exact payload for the next request."""
        head: list[Message] = []
        if self.system:
            head.append(Message(role="system", content=self.system))
        kept = self.history[-(self.max_turns * 2) :]
        return head + kept

    def ask(self, text: str) -> Completion:
        self.history.append(Message(role="user", content=text))
        result = self.client.complete(self.messages())
        self.history.append(
            Message(role="assistant", content=result.text)
        )
        return result

    def turns(self) -> int:
        return len(self.history) // 2
