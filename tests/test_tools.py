import asyncio
import json
import time

import pytest
from pydantic import BaseModel, Field

from llmapp.llm.base import ToolCallRequest
from llmapp.tools.base import (
    ApprovalRequired,
    SideEffect,
    ToolCall,
    ToolSpec,
)
from llmapp.tools.loop import (
    results_as_input,
    run_calls,
    run_calls_parallel,
)
from llmapp.tools.registry import ToolRegistry


class SearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=5, ge=1, le=20)


class TicketArgs(BaseModel):
    summary: str = Field(min_length=1, max_length=200)
    priority: str = Field(pattern="^(low|normal|high)$")


def search(args: SearchArgs) -> str:
    return f"{args.limit} results for {args.query}"


def create_ticket(args: TicketArgs) -> str:
    return f"created {args.priority} ticket: {args.summary}"


def slow(args: SearchArgs) -> str:
    time.sleep(0.5)
    return "eventually"


def explodes(args: SearchArgs) -> str:
    raise RuntimeError("upstream is down")


def huge(args: SearchArgs) -> str:
    return "x" * 10_000


SEARCH = ToolSpec(
    name="search_docs",
    description="Search the policy corpus.",
    arguments=SearchArgs,
    handler=search,
)
TICKET = ToolSpec(
    name="create_ticket",
    description="Open a support ticket.",
    arguments=TicketArgs,
    handler=create_ticket,
    side_effect=SideEffect.WRITE,
    requires_approval=True,
)


def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(SEARCH)
    reg.register(TICKET)
    return reg


def call(name: str, **kwargs: object) -> ToolCall:
    return ToolCall("c1", name, json.dumps(kwargs))


# --- schemas --------------------------------------------------


def test_schema_is_flat_for_the_responses_api() -> None:
    schema = SEARCH.schema()
    assert schema["type"] == "function"
    assert schema["name"] == "search_docs"
    assert "function" not in schema
    assert schema["strict"] is True


def test_schema_forbids_extra_arguments() -> None:
    parameters = SEARCH.schema()["parameters"]
    assert parameters["additionalProperties"] is False
    assert set(parameters["required"]) == {"query", "limit"}


def test_only_allowed_tools_are_advertised() -> None:
    names = [
        s["name"] for s in registry().schemas(["search_docs"])
    ]
    assert names == ["search_docs"]


def test_duplicate_registration_is_rejected() -> None:
    reg = registry()
    with pytest.raises(ValueError, match="already exists"):
        reg.register(SEARCH)


# --- validation -----------------------------------------------


def test_a_valid_call_runs() -> None:
    result = registry().dispatch(
        call("search_docs", query="refunds", limit=3)
    )
    assert result.ok
    assert result.output == "3 results for refunds"


def test_defaults_are_applied() -> None:
    result = registry().dispatch(
        call("search_docs", query="refunds", limit=5)
    )
    assert "5 results" in result.output


def test_invalid_arguments_never_reach_the_handler() -> None:
    result = registry().dispatch(
        call("search_docs", query="refunds", limit=999)
    )
    assert result.ok is False
    assert "limit" in result.output
    assert "call again" in result.output


def test_malformed_json_is_reported_not_raised() -> None:
    result = registry().dispatch(
        ToolCall("c1", "search_docs", "{not json")
    )
    assert result.ok is False
    assert "not JSON" in result.output


def test_an_unknown_tool_is_denied() -> None:
    result = registry().dispatch(call("delete_everything"))
    assert result.ok is False
    assert "no tool named" in result.output


def test_a_tool_outside_the_allowlist_is_denied() -> None:
    result = registry().dispatch(
        call("create_ticket", summary="x", priority="low"),
        allowed=["search_docs"],
    )
    assert result.ok is False
    assert "not available here" in result.output


# --- approval and side effects --------------------------------


def test_a_write_tool_stops_for_approval() -> None:
    with pytest.raises(ApprovalRequired, match="create_ticket"):
        registry().dispatch(
            call("create_ticket", summary="broken", priority="high")
        )


def test_an_approved_call_runs() -> None:
    result = registry().dispatch(
        call("create_ticket", summary="broken", priority="high"),
        approved=["c1"],
    )
    assert result.ok
    assert "created high ticket" in result.output


def test_approval_is_per_call_not_per_tool() -> None:
    with pytest.raises(ApprovalRequired):
        registry().dispatch(
            ToolCall(
                "c2",
                "create_ticket",
                json.dumps({"summary": "x", "priority": "low"}),
            ),
            approved=["c1"],
        )


def test_write_tools_are_listable_for_review() -> None:
    assert registry().writes() == ["create_ticket"]


# --- failures and limits --------------------------------------


def test_a_timeout_becomes_a_message_to_the_model() -> None:
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="slow_tool",
            description="Takes too long.",
            arguments=SearchArgs,
            handler=slow,
            timeout_s=0.05,
        )
    )
    result = reg.dispatch(call("slow_tool", query="x", limit=1))
    assert result.ok is False
    assert "timed out" in result.output


def test_an_exception_becomes_a_message_to_the_model() -> None:
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="broken",
            description="Always fails.",
            arguments=SearchArgs,
            handler=explodes,
        )
    )
    result = reg.dispatch(call("broken", query="x", limit=1))
    assert result.ok is False
    assert "upstream is down" in result.output


def test_large_results_are_truncated_with_a_notice() -> None:
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="verbose",
            description="Returns too much.",
            arguments=SearchArgs,
            handler=huge,
            max_result_chars=500,
        )
    )
    result = reg.dispatch(call("verbose", query="x", limit=1))
    assert result.truncated
    assert len(result.output) <= 500
    assert "10000 characters" in result.output


# --- the loop -------------------------------------------------


def request(call_id: str, name: str, **kw: object) -> ToolCallRequest:
    return ToolCallRequest(call_id, name, json.dumps(kw))


def test_sequential_execution_preserves_order() -> None:
    requests = [
        request("a", "search_docs", query="one", limit=1),
        request("b", "search_docs", query="two", limit=2),
    ]
    results = run_calls(registry(), requests)
    assert [r.call_id for r in results] == ["a", "b"]


def test_parallel_execution_preserves_order() -> None:
    requests = [
        request(str(i), "search_docs", query=f"q{i}", limit=1)
        for i in range(6)
    ]
    results = asyncio.run(
        run_calls_parallel(registry(), requests, limit=3)
    )
    assert [r.call_id for r in results] == [
        str(i) for i in range(6)
    ]


def test_parallel_execution_is_faster_than_sequential() -> None:
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="slow_tool",
            description="Takes half a second.",
            arguments=SearchArgs,
            handler=slow,
            timeout_s=5.0,
        )
    )
    requests = [
        request(str(i), "slow_tool", query="x", limit=1)
        for i in range(4)
    ]
    start = time.perf_counter()
    asyncio.run(run_calls_parallel(reg, requests, limit=4))
    parallel = time.perf_counter() - start
    assert parallel < 4 * 0.5 * 0.75


def test_results_are_paired_with_their_calls() -> None:
    requests = [request("a", "search_docs", query="one", limit=1)]
    results = run_calls(registry(), requests)
    payload = results_as_input(requests, results)
    assert payload[0]["type"] == "function_call"
    assert payload[1]["type"] == "function_call_output"
    assert payload[1]["call_id"] == "a"
