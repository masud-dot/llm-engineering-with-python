import json
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from llmapp.agents.loop import Agent, approve
from llmapp.agents.state import (
    AgentState,
    Limits,
    RunStatus,
    signature,
)
from llmapp.llm.base import Completion, Message, ToolCallRequest
from llmapp.prompts.registry import PromptRegistry
from llmapp.tools.base import SideEffect, ToolSpec
from llmapp.tools.registry import ToolRegistry

REGISTRY = PromptRegistry(Path("prompts"))


class SearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=200)


class PublishArgs(BaseModel):
    title: str = Field(min_length=1, max_length=100)


FACTS = {
    "refund window": "Refunds are issued within fourteen days.",
    "escalation": "Unresolved tickets go to a senior engineer.",
}


def search(args: SearchArgs) -> str:
    return FACTS.get(args.query, "no results")


def publish(args: PublishArgs) -> str:
    return f"published: {args.title}"


def tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="search",
            description="Search the corpus.",
            arguments=SearchArgs,
            handler=search,
        )
    )
    reg.register(
        ToolSpec(
            name="publish",
            description="Publish a brief.",
            arguments=PublishArgs,
            handler=publish,
            side_effect=SideEffect.WRITE,
            requires_approval=True,
        )
    )
    return reg


class Script:
    """Return prepared turns: text, or tool calls."""

    def __init__(self, *turns: object) -> None:
        self.turns = list(turns)
        self.calls: list[list[Message]] = []
        self.tools_seen: list[list[str]] = []

    def _next(self, messages: object) -> object:
        self.calls.append(list(messages))  # type: ignore[arg-type]
        index = len(self.calls) - 1
        if index >= len(self.turns):
            return "I have run out of script."
        return self.turns[index]

    def complete(
        self,
        messages: object,
        *,
        max_output_tokens: int = 512,
    ) -> Completion:
        turn = self._next(messages)
        return self._build(turn)

    def complete_with_tools(
        self,
        messages: object,
        *,
        tools: object = (),
        extra_input: object = (),
        max_output_tokens: int = 512,
    ) -> Completion:
        self.tools_seen.append(
            [t["name"] for t in tools]  # type: ignore[index,union-attr]
        )
        return self._build(self._next(messages))

    @staticmethod
    def _build(turn: object) -> Completion:
        if isinstance(turn, str):
            return Completion(turn, 40, 20, "script", "stop")
        calls = tuple(
            ToolCallRequest(cid, name, json.dumps(args))
            for cid, name, args in turn  # type: ignore[union-attr]
        )
        return Completion("", 40, 20, "script", "stop", calls)


def call(cid: str, name: str, **kw: object) -> tuple[object, ...]:
    return (cid, name, kw)


# --- the loop -------------------------------------------------


def test_a_plan_is_written_before_acting() -> None:
    client = Script(
        "search for the policy\nsummarize the result",
        "Refunds take fourteen days.",
    )
    agent = Agent(client, tools(), REGISTRY)
    state = agent.run("what is the refund window")
    assert state.plan == [
        "search for the policy",
        "summarize the result",
    ]


def test_a_tool_call_then_an_answer() -> None:
    client = Script(
        "search then answer",
        [call("c1", "search", query="refund window")],
        "Refunds are issued within fourteen days.",
    )
    state = Agent(client, tools(), REGISTRY).run("refund window")
    assert state.status is RunStatus.COMPLETED
    assert "fourteen days" in state.answer
    assert state.tool_calls == 1


def test_tool_results_reach_the_next_prompt() -> None:
    client = Script(
        "plan",
        [call("c1", "search", query="refund window")],
        "done",
    )
    Agent(client, tools(), REGISTRY).run("refund window")
    third = client.calls[2][1].content
    assert "fourteen days" in third


def test_only_allowed_tools_are_offered() -> None:
    client = Script("plan", "done")
    Agent(
        client, tools(), REGISTRY, allowed=["search"]
    ).run("anything")
    assert client.tools_seen == [["search"]]


# --- termination ----------------------------------------------


def test_the_step_limit_stops_a_runaway() -> None:
    turns = ["plan"] + [
        [call(f"c{i}", "search", query=f"topic {i}")]
        for i in range(20)
    ]
    client = Script(*turns)
    state = Agent(client, tools(), REGISTRY).run(
        "keep going", Limits(max_steps=3)
    )
    assert state.status is RunStatus.STEP_LIMIT
    assert state.step_count == 3


def test_the_tool_call_limit_stops_a_runaway() -> None:
    turns = ["plan"] + [
        [call(f"c{i}", "search", query=f"topic {i}")]
        for i in range(20)
    ]
    state = Agent(Script(*turns), tools(), REGISTRY).run(
        "keep going", Limits(max_steps=50, max_tool_calls=4)
    )
    assert state.status is RunStatus.STEP_LIMIT
    assert state.tool_calls >= 4


def test_an_identical_repeated_call_is_a_stall() -> None:
    same = [call("c1", "search", query="refund window")]
    client = Script("plan", same, same, same, same)
    state = Agent(client, tools(), REGISTRY).run(
        "refund window", Limits(max_steps=10)
    )
    assert state.status is RunStatus.STALLED
    assert "repeatedly" in state.note


def test_the_nudge_is_sent_before_giving_up() -> None:
    same = [call("c1", "search", query="refund window")]
    client = Script("plan", same, same, same, same)
    Agent(client, tools(), REGISTRY).run(
        "refund window", Limits(max_steps=10)
    )
    nudged = [
        turn
        for turn in client.calls
        if any("already made that exact call" in m.content
               for m in turn)
    ]
    assert nudged


def test_different_arguments_are_not_a_repeat() -> None:
    a = ToolCallRequest("c1", "search", '{"query": "one"}')
    b = ToolCallRequest("c2", "search", '{"query": "two"}')
    assert signature(a) != signature(b)


def test_the_call_id_does_not_affect_the_signature() -> None:
    a = ToolCallRequest("c1", "search", '{"query": "one"}')
    b = ToolCallRequest("c9", "search", '{"query": "one"}')
    assert signature(a) == signature(b)


# --- approval -------------------------------------------------


def test_a_write_tool_suspends_the_run() -> None:
    client = Script(
        "plan",
        [call("w1", "publish", title="Refund brief")],
        "published",
    )
    state = Agent(client, tools(), REGISTRY).run("publish it")
    assert state.status is RunStatus.NEEDS_APPROVAL
    assert state.pending is not None
    assert state.pending.name == "publish"


def test_an_approved_run_resumes_and_completes() -> None:
    client = Script(
        "plan",
        [call("w1", "publish", title="Refund brief")],
        "the brief is published",
    )
    agent = Agent(client, tools(), REGISTRY)
    state = agent.run("publish it")
    assert state.status is RunStatus.NEEDS_APPROVAL

    approve(state, "w1")
    resumed = agent.resume(state)
    assert resumed.status is RunStatus.COMPLETED
    assert "published" in resumed.answer


def test_declining_leaves_the_run_suspended() -> None:
    client = Script(
        "plan", [call("w1", "publish", title="x")], "unused"
    )
    state = Agent(client, tools(), REGISTRY).run("publish it")
    assert state.status is RunStatus.NEEDS_APPROVAL
    assert state.approved == set()


# --- state ----------------------------------------------------


def test_status_terminality() -> None:
    assert RunStatus.COMPLETED.is_terminal
    assert RunStatus.STEP_LIMIT.is_terminal
    assert not RunStatus.NEEDS_APPROVAL.is_terminal
    assert not RunStatus.RUNNING.is_terminal


def test_the_transcript_records_calls_and_results() -> None:
    client = Script(
        "plan",
        [call("c1", "search", query="escalation")],
        "done",
    )
    state = Agent(client, tools(), REGISTRY).run("escalation")
    text = state.transcript()
    assert "called search" in text
    assert "senior engineer" in text


def test_a_failed_tool_does_not_end_the_run() -> None:
    client = Script(
        "plan",
        [call("c1", "search", query="")],
        "I could not find it.",
    )
    state = Agent(client, tools(), REGISTRY).run("bad query")
    assert state.status is RunStatus.COMPLETED
    assert not state.steps[0].results[0].ok


def test_cost_limits_are_checked() -> None:
    state = AgentState(
        goal="x", limits=Limits(max_cost_usd=0.01)
    )
    state.spent_usd = 0.02
    assert state.exhausted() is RunStatus.COST_LIMIT
