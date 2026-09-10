"""A scripted client used for tests and local runs."""

import asyncio
from collections.abc import Iterator, Sequence

from llmapp.llm.base import Completion, Message


class ScriptedClient:
    """Return prepared replies in order, recording calls."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self._index = 0
        self.calls: list[list[Message]] = []

    def _next(self, messages: Sequence[Message]) -> Completion:
        self.calls.append(list(messages))
        if self._index >= len(self._replies):
            raise AssertionError("more calls than replies")
        text = self._replies[self._index]
        self._index += 1
        prompt = " ".join(m.content for m in messages)
        return Completion(
            text=text,
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
            model="scripted",
            finish_reason="stop",
        )

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_output_tokens: int = 512,
    ) -> Completion:
        return self._next(messages)

    def stream(
        self,
        messages: Sequence[Message],
        *,
        max_output_tokens: int = 512,
    ) -> Iterator[str]:
        for word in self._next(messages).text.split(" "):
            yield word + " "

    async def acomplete(
        self,
        messages: Sequence[Message],
        *,
        max_output_tokens: int = 512,
    ) -> Completion:
        await asyncio.sleep(0)
        return self._next(messages)
