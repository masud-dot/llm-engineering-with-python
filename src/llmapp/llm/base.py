"""The model-access port.

This module defines what the application needs from a
language model. It imports no provider SDK, so nothing
below it can leak a vendor type into the layers above.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    """One turn of a conversation."""

    role: Role
    content: str


@dataclass(frozen=True)
class ToolCallRequest:
    """A tool the model wants run, as it came off the wire."""

    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Completion:
    """One model response and what it cost."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str = ""
    finish_reason: str = ""
    tool_calls: tuple[ToolCallRequest, ...] = ()


class LLMClient(Protocol):
    """Everything the application asks of a model."""

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_output_tokens: int = 512,
    ) -> Completion: ...


@runtime_checkable
class SupportsStreaming(Protocol):
    """A client that can yield text as it is produced."""

    def stream(
        self,
        messages: Sequence[Message],
        *,
        max_output_tokens: int = 512,
    ) -> Iterator[str]: ...


@runtime_checkable
class SupportsTools(Protocol):
    """A client that can be given tools to call."""

    def complete_with_tools(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, object]],
        extra_input: Sequence[dict[str, object]] = (),
        max_output_tokens: int = 512,
    ) -> Completion: ...


@runtime_checkable
class SupportsStructuredOutput(Protocol):
    """A client that can constrain output to a JSON Schema."""

    def complete_json(
        self,
        messages: Sequence[Message],
        *,
        schema: dict[str, object],
        schema_name: str,
        max_output_tokens: int = 512,
    ) -> Completion: ...


@runtime_checkable
class SupportsAsync(Protocol):
    """A client that can be awaited."""

    async def acomplete(
        self,
        messages: Sequence[Message],
        *,
        max_output_tokens: int = 512,
    ) -> Completion: ...
