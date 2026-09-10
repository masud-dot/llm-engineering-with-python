"""Project 4: a research agent with bounded autonomy."""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from llmapp.agents.loop import Agent
from llmapp.agents.state import Limits
from llmapp.llm.base import LLMClient
from llmapp.prompts.registry import PromptRegistry
from llmapp.retrieval.search import SemanticIndex
from llmapp.tools.base import SideEffect, ToolSpec
from llmapp.tools.registry import ToolRegistry

MAX_SNIPPET = 300


class SearchArgs(BaseModel):
    """Arguments for the corpus search tool."""

    query: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=3, ge=1, le=5)


class PublishArgs(BaseModel):
    """Arguments for the brief-publishing tool."""

    title: str = Field(min_length=1, max_length=120)
    body: str = Field(min_length=1, max_length=4000)


@dataclass
class Notebook:
    """Where a published brief goes. Replace in production."""

    briefs: dict[str, str] = field(default_factory=dict)

    def publish(self, title: str, body: str) -> str:
        if title in self.briefs:  # idempotent by title
            return f"already published: {title}"
        self.briefs[title] = body
        return f"published: {title}"


def build_research_agent(
    client: LLMClient,
    index: SemanticIndex,
    registry: PromptRegistry,
    notebook: Notebook,
    *,
    limits: Limits | None = None,
) -> tuple[Agent, Limits]:
    """Two tools: one reads, one writes and needs approval."""

    def search(args: SearchArgs) -> str:
        hits = index.search(args.query, k=args.limit)
        if not hits:
            return "no results"
        return "\n".join(
            f"[{h.chunk.chunk_id}] {h.chunk.text[:MAX_SNIPPET]}"
            for h in hits
        )

    def publish(args: PublishArgs) -> str:
        return notebook.publish(args.title, args.body)

    tools = ToolRegistry()
    tools.register(
        ToolSpec(
            name="search_corpus",
            description=(
                "Search the internal document corpus. Use for "
                "questions about company policy and procedure."
            ),
            arguments=SearchArgs,
            handler=search,
            side_effect=SideEffect.READ,
            timeout_s=5.0,
        )
    )
    tools.register(
        ToolSpec(
            name="publish_brief",
            description=(
                "Publish a finished research brief. Use only "
                "once, after gathering evidence."
            ),
            arguments=PublishArgs,
            handler=publish,
            side_effect=SideEffect.WRITE,
            requires_approval=True,
            timeout_s=5.0,
        )
    )
    agent = Agent(client=client, tools=tools, registry=registry)
    return agent, limits or Limits(max_steps=6, max_tool_calls=8)


def summarize(state: Any) -> dict[str, object]:
    """The audit record for one research run."""
    return {
        "goal": state.goal,
        "status": state.status.value,
        "plan": state.plan,
        "steps": state.step_count,
        "tool_calls": state.tool_calls,
        "answer": state.answer,
        "note": state.note,
    }
