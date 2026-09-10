from pathlib import Path

from llmapp.llm.fake import ScriptedClient
from llmapp.prompts.chain import TriageChain
from llmapp.prompts.registry import PromptRegistry

CATEGORIES = "billing, shipping, technical, other"
FACTS = (
    "mentions_charge: yes\n"
    "mentions_delivery: yes\n"
    "mentions_error: no"
)


def build() -> tuple[TriageChain, ScriptedClient]:
    client = ScriptedClient([FACTS, "billing"])
    chain = TriageChain(
        client, PromptRegistry(Path("prompts")), CATEGORIES
    )
    return chain, client


def test_chain_returns_the_second_step_answer() -> None:
    chain, _ = build()
    result = chain.run("Charged for express that never arrived.")
    assert result.category == "billing"


def test_step_two_sees_facts_and_not_the_raw_ticket() -> None:
    chain, client = build()
    chain.run("Charged for express that never arrived.")
    second = client.calls[1]
    body = second[1].content
    assert "mentions_charge: yes" in body
    assert "express" not in body


def test_the_intermediate_artifact_is_returned() -> None:
    chain, _ = build()
    result = chain.run("anything")
    assert result.facts == FACTS
    assert result.prompt_ids[0].startswith("extract.v1.")
    assert result.prompt_ids[1].startswith("judge.v1.")
