import json
from pathlib import Path

import pytest

from llmapp.llm.base import Completion, Message
from llmapp.prompts.context import ContextBudget
from llmapp.prompts.registry import PromptRegistry
from llmapp.rag.pipeline import RagPipeline, build_index_from
from llmapp.retrieval.search import SemanticIndex
from tests._fake_embedder import KeywordEmbedder

CORPUS = Path("tests/corpus")
REGISTRY = PromptRegistry(Path("prompts"))


class Replies:
    """Return canned JSON, recording what was sent."""

    def __init__(self, *payloads: object) -> None:
        self.payloads = list(payloads)
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: object,
        *,
        max_output_tokens: int = 512,
    ) -> Completion:
        self.calls.append(list(messages))  # type: ignore[arg-type]
        payload = self.payloads[len(self.calls) - 1]
        text = (
            payload
            if isinstance(payload, str)
            else json.dumps(payload)
        )
        return Completion(text, 100, 40, "fake", "stop")


@pytest.fixture
def index() -> SemanticIndex:
    built = SemanticIndex(KeywordEmbedder())
    build_index_from(CORPUS, built, max_words=60)
    return built


def chunk_id_for(index: SemanticIndex, query: str) -> str:
    return index.search(query, k=1)[0].chunk.chunk_id


def test_the_index_is_populated(index: SemanticIndex) -> None:
    assert len(index) > 3


def test_retrieval_finds_the_right_document(
    index: SemanticIndex,
) -> None:
    hits = index.search("refund window fourteen days", k=1)
    assert hits[0].chunk.source == "refunds.md"


def test_answer_carries_verified_citations(
    index: SemanticIndex,
) -> None:
    target = chunk_id_for(index, "refund window fourteen days")
    client = Replies(
        {
            "answered": True,
            "answer": "Refunds are issued within fourteen days.",
            "citations": [
                {
                    "chunk_id": target,
                    "quote": "Refunds are issued within fourteen",
                }
            ],
            "missing": "",
        }
    )
    pipeline = RagPipeline(client, index, REGISTRY)
    result = pipeline.answer("what is the refund window")
    assert result.answer.answered is True
    assert result.answer.citations[0].chunk_id == target
    assert "refunds.md" in result.sources
    assert result.prompt_id.startswith("rag_answer.v1.")


def test_passages_and_ids_reach_the_prompt(
    index: SemanticIndex,
) -> None:
    client = Replies({"answered": False, "missing": "n/a"})
    RagPipeline(client, index, REGISTRY).answer("refund window")
    body = client.calls[0][1].content
    assert "BEGIN PASSAGES" in body
    assert "refunds.md#" in body


def test_a_fabricated_citation_forces_a_refusal(
    index: SemanticIndex,
) -> None:
    client = Replies(
        {
            "answered": True,
            "answer": "Refunds take ninety days.",
            "citations": [
                {"chunk_id": "invented.md#7", "quote": "ninety"}
            ],
            "missing": "",
        }
    )
    result = RagPipeline(client, index, REGISTRY).answer(
        "what is the refund window"
    )
    assert result.answer.answered is False
    assert "traced" in result.answer.missing


def test_a_quote_that_is_not_in_the_passage_is_rejected(
    index: SemanticIndex,
) -> None:
    target = chunk_id_for(index, "refund window fourteen days")
    client = Replies(
        {
            "answered": True,
            "answer": "Refunds take ninety days.",
            "citations": [
                {"chunk_id": target, "quote": "ninety days"}
            ],
            "missing": "",
        }
    )
    result = RagPipeline(client, index, REGISTRY).answer(
        "what is the refund window"
    )
    assert result.answer.answered is False


def test_nothing_relevant_means_no_model_call(
    index: SemanticIndex,
) -> None:
    client = Replies()
    result = RagPipeline(client, index, REGISTRY).answer(
        "quarterly revenue in Luxembourg"
    )
    assert result.answer.answered is False
    assert result.hits == []
    assert client.calls == []


def test_the_model_may_refuse_and_say_why(
    index: SemanticIndex,
) -> None:
    client = Replies(
        {
            "answered": False,
            "answer": "",
            "citations": [],
            "missing": "no passage states the fee amount",
        }
    )
    result = RagPipeline(client, index, REGISTRY).answer(
        "express delivery surcharge"
    )
    assert result.answer.answered is False
    assert "fee amount" in result.answer.missing


def test_a_tight_budget_drops_the_weakest_passages(
    index: SemanticIndex,
) -> None:
    client = Replies({"answered": False, "missing": "n/a"})
    pipeline = RagPipeline(
        client,
        index,
        REGISTRY,
        budget=ContextBudget(
            window_tokens=200, reserved_output=60
        ),
        k=5,
    )
    result = pipeline.answer("refund delivery escalation policy")
    assert result.dropped or len(result.hits) < 5
