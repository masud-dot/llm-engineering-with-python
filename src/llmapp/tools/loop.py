"""Run tool calls, including several at once."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from llmapp.llm.base import Completion, Message, ToolCallRequest
from llmapp.tools.base import ToolCall, ToolResult
from llmapp.tools.registry import ToolRegistry


@dataclass(frozen=True)
class ToolTurn:
    """One round of calls and their results."""

    completion: Completion
    results: list[ToolResult]

    @property
    def failed(self) -> list[ToolResult]:
        return [r for r in self.results if not r.ok]


def to_call(request: ToolCallRequest) -> ToolCall:
    return ToolCall(
        call_id=request.call_id,
        name=request.name,
        arguments=request.arguments,
    )


def run_calls(
    registry: ToolRegistry,
    requests: Sequence[ToolCallRequest],
    *,
    allowed: Sequence[str] | None = None,
    approved: Sequence[str] = (),
) -> list[ToolResult]:
    """Sequential execution, in the order requested."""
    return [
        registry.dispatch(
            to_call(request), allowed=allowed, approved=approved
        )
        for request in requests
    ]


async def run_calls_parallel(
    registry: ToolRegistry,
    requests: Sequence[ToolCallRequest],
    *,
    allowed: Sequence[str] | None = None,
    approved: Sequence[str] = (),
    limit: int = 4,
) -> list[ToolResult]:
    """Independent calls run together, bounded.

    Order is preserved so the results line up with the calls
    the model made, whatever order they finish in.
    """
    semaphore = asyncio.Semaphore(limit)

    async def one(request: ToolCallRequest) -> ToolResult:
        async with semaphore:
            return await asyncio.to_thread(
                registry.dispatch,
                to_call(request),
                allowed=allowed,
                approved=approved,
            )

    return list(
        await asyncio.gather(*(one(r) for r in requests))
    )


def results_as_input(
    requests: Sequence[ToolCallRequest],
    results: Sequence[ToolResult],
) -> list[dict[str, object]]:
    """The call and its output, in the order the API expects."""
    payload: list[dict[str, object]] = []
    for request, result in zip(requests, results, strict=True):
        payload.append(
            {
                "type": "function_call",
                "call_id": request.call_id,
                "name": request.name,
                "arguments": request.arguments,
            }
        )
        payload.append(dict(result.as_input()))
    return payload


def conversation_with_results(
    messages: Sequence[Message],
    requests: Sequence[ToolCallRequest],
    results: Sequence[ToolResult],
) -> tuple[list[Message], list[dict[str, object]]]:
    """Messages plus the tool exchange, ready to send back."""
    return list(messages), results_as_input(requests, results)
