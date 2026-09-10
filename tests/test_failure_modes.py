"""Reproduce each RAG failure mode, then show the response."""

import json
from pathlib import Path

from llmapp.llm.base import Completion, Message
from llmapp.prompts.context import ContextBudget
from llmapp.prompts.registry import PromptRegistry
from llmapp.rag.pipeline import RagPipeline
from llmapp.retrieval.chunking import Chunk
from llmapp.retrieval.search import SemanticIndex
from tests._fake_embedder import KeywordEmbedder

REGISTRY = PromptRegistry(Path("prompts"))

CURRENT = Chunk(
    "Refunds are issued within fourteen days of the return.",
    "refunds.md", 0,
    {"topic": "refunds", "effective_date": "2026-01-01"},
)
SUPERSEDED = Chunk(
    "Refunds are issued within ninety days of the return.",
    "refunds-2024.md", 0,
    {"topic": "refunds", "effective_date": "2024-01-01"},
)
NOISE = Chunk(
    "Escalation passes tickets to a senior engineer.",
    "escalation.md", 0, {"topic": "escalation"},
)


class Replies:
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
        return Completion(text, 80, 30, "fake", "stop")


def index_of(*chunks: Chunk) -> SemanticIndex:
    built = SemanticIndex(KeywordEmbedder())
    built.add(list(chunks))
    return built


def test_1_missed_retrieval_produces_a_refusal() -> None:
    """The corpus cannot answer, so nothing clears the floor."""
    client = Replies()
    result = RagPipeline(
        client, index_of(NOISE), REGISTRY
    ).answer("quarterly revenue in Luxembourg")
    assert result.answer.answered is False
    assert client.calls == []


def test_2_distraction_is_limited_by_the_relevance_floor(
) -> None:
    """Weak passages are excluded before the model sees them."""
    result_hits = index_of(CURRENT, NOISE).search(
        "refund fourteen days", k=5
    )
    kept = [h for h in result_hits if h.score >= 0.20]
    assert [h.chunk.source for h in kept] == ["refunds.md"]


def test_3_budget_drops_are_recorded_not_silent() -> None:
    client = Replies({"answered": False, "missing": "n/a"})
    pipeline = RagPipeline(
        client, index_of(CURRENT, SUPERSEDED, NOISE), REGISTRY,
        budget=ContextBudget(
            window_tokens=80, reserved_output=40
        ),
        k=3, min_score=0.0,
    )
    result = pipeline.answer("refunds escalation days")
    assert result.dropped or result.truncated


def test_4_ignored_evidence_is_caught_by_coverage() -> None:
    """The passage says fourteen; the answer says ninety."""
    client = Replies(
        {
            "answered": True,
            "answer": "Refunds are issued within ninety days.",
            "citations": [
                {
                    "chunk_id": "refunds.md#0",
                    "quote": "Refunds are issued within",
                }
            ],
            "missing": "",
        }
    )
    result = RagPipeline(
        client, index_of(CURRENT), REGISTRY
    ).answer("refunds fourteen days")
    assert result.answer.answered is False


def test_5_a_fabricated_citation_is_rejected() -> None:
    client = Replies(
        {
            "answered": True,
            "answer": "Refunds take fourteen days.",
            "citations": [
                {"chunk_id": "invented.md#3", "quote": "fourteen"}
            ],
            "missing": "",
        }
    )
    result = RagPipeline(
        client, index_of(CURRENT), REGISTRY
    ).answer("refunds fourteen days")
    assert result.answer.answered is False
    assert "traced" in result.answer.missing


def test_6_a_superseded_source_is_flagged_as_a_conflict(
) -> None:
    client = Replies(
        {
            "answered": True,
            "answer": "Refunds are issued within ninety days.",
            "citations": [
                {
                    "chunk_id": "refunds-2024.md#0",
                    "quote": "ninety days",
                }
            ],
            "missing": "",
        }
    )
    result = RagPipeline(
        client, index_of(CURRENT, SUPERSEDED), REGISTRY,
        min_score=0.0,
    ).answer("refunds days")
    assert result.grounding is not None
    assert result.grounding.conflicts == ["refunds"]


def test_6b_a_date_filter_removes_the_superseded_source(
) -> None:
    client = Replies({"answered": False, "missing": "n/a"})
    result = RagPipeline(
        client, index_of(CURRENT, SUPERSEDED), REGISTRY,
        min_score=0.0,
    ).answer(
        "refunds days", where={"effective_date": "2026-01-01"}
    )
    assert result.sources == ["refunds.md"]
