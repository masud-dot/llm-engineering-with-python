import json
from pathlib import Path

from llmapp.agents.loop import Agent, approve
from llmapp.agents.research import (
    Notebook,
    build_research_agent,
    summarize,
)
from llmapp.agents.state import RunStatus
from llmapp.prompts.registry import PromptRegistry
from llmapp.rag.pipeline import build_index_from
from llmapp.retrieval.search import SemanticIndex
from tests._fake_embedder import KeywordEmbedder
from tests.test_agent import Script, call

REGISTRY = PromptRegistry(Path("prompts"))


def build(*turns: object) -> tuple[Agent, Notebook, object]:
    index = SemanticIndex(KeywordEmbedder())
    build_index_from(Path("tests/corpus"), index, max_words=60)
    notebook = Notebook()
    agent, limits = build_research_agent(
        Script(*turns), index, REGISTRY, notebook
    )
    return agent, notebook, limits


def test_the_agent_searches_then_publishes_with_approval(
) -> None:
    agent, notebook, limits = build(
        "search the corpus\nwrite the brief\npublish it",
        [call("c1", "search_corpus", query="refunds days")],
        [
            call(
                "w1",
                "publish_brief",
                title="Refund policy",
                body="Refunds are issued within fourteen days.",
            )
        ],
        "The brief has been published.",
    )
    state = agent.run("summarize the refund policy", limits)  # type: ignore[arg-type]
    assert state.status is RunStatus.NEEDS_APPROVAL
    assert notebook.briefs == {}

    approve(state, "w1")
    final = agent.resume(state)
    assert final.status is RunStatus.COMPLETED
    assert "Refund policy" in notebook.briefs


def test_search_results_carry_chunk_ids() -> None:
    agent, _, limits = build(
        "plan",
        [call("c1", "search_corpus", query="refunds days")],
        "done",
    )
    state = agent.run("refund policy", limits)  # type: ignore[arg-type]
    assert "refunds.md#0" in state.steps[0].results[0].output


def test_publishing_twice_is_idempotent() -> None:
    notebook = Notebook()
    assert "published" in notebook.publish("t", "b")
    assert "already published" in notebook.publish("t", "b")
    assert len(notebook.briefs) == 1


def test_the_run_summary_is_auditable() -> None:
    agent, _, limits = build(
        "plan",
        [call("c1", "search_corpus", query="refunds days")],
        "Refunds take fourteen days.",
    )
    state = agent.run("refund policy", limits)  # type: ignore[arg-type]
    record = summarize(state)
    assert record["status"] == "completed"
    assert record["tool_calls"] == 1
    assert json.dumps(record)


def test_the_approved_call_is_the_one_that_runs() -> None:
    """Approval binds to the call the human saw, not a new one."""
    agent, notebook, limits = build(
        "plan",
        [
            call(
                "w1",
                "publish_brief",
                title="Refund policy",
                body="Refunds are issued within fourteen days.",
            )
        ],
        # If the loop re-asked the model, it would get this
        # instead, and publish something different.
        [
            call(
                "w2",
                "publish_brief",
                title="Something else entirely",
                body="Not what was approved.",
            )
        ],
        "done",
    )
    state = agent.run("summarize the refund policy", limits)  # type: ignore[arg-type]
    approve(state, "w1")
    agent.resume(state)
    assert "Refund policy" in notebook.briefs
    assert "Something else entirely" not in notebook.briefs
