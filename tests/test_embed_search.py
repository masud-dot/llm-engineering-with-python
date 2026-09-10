import numpy as np
import pytest

from llmapp.retrieval.chunking import Chunk
from llmapp.retrieval.embed import OpenAIEmbedder
from llmapp.retrieval.metrics import (
    evaluate,
    recall_at_k,
    reciprocal_rank,
)
from llmapp.retrieval.search import SemanticIndex
from tests._mock_embeddings import SEEN, serve


@pytest.fixture(scope="module")
def embedder() -> OpenAIEmbedder:
    serve(8093)
    return OpenAIEmbedder(
        api_key="test",
        model="mock-embed",
        dimensions=8,
        base_url="http://127.0.0.1:8093/v1",
        batch_size=2,
    )


@pytest.mark.local_server
def test_requests_are_batched(embedder: OpenAIEmbedder) -> None:
    SEEN.clear()
    texts = [f"document number {i}" for i in range(5)]
    matrix = embedder.embed(texts)
    assert matrix.shape == (5, 8)
    assert len(SEEN) == 3  # batch_size=2 over 5 unique texts


@pytest.mark.local_server
def test_the_cache_prevents_re_embedding(
    embedder: OpenAIEmbedder,
) -> None:
    SEEN.clear()
    embedder.embed(["a repeated policy sentence"])
    first = len(SEEN)
    embedder.embed(["a repeated policy sentence"])
    assert len(SEEN) == first


@pytest.mark.local_server
def test_dimensions_are_requested_and_checked(
    embedder: OpenAIEmbedder,
) -> None:
    SEEN.clear()
    embedder.embed(["dimension check"])
    assert SEEN[0]["dimensions"] == 8


@pytest.mark.local_server
def test_search_returns_the_matching_chunk(
    embedder: OpenAIEmbedder,
) -> None:
    index = SemanticIndex(embedder)
    chunks = [
        Chunk("refund policy text", "a.md", 0, {"team": "billing"}),
        Chunk("delivery policy text", "b.md", 0, {"team": "ship"}),
    ]
    index.add(chunks)
    hits = index.search("refund policy text", k=1)
    assert len(hits) == 1
    assert hits[0].chunk.chunk_id == "a.md#0"
    assert hits[0].score == pytest.approx(1.0)


@pytest.mark.local_server
def test_metadata_filters_before_ranking(
    embedder: OpenAIEmbedder,
) -> None:
    index = SemanticIndex(embedder)
    index.add(
        [
            Chunk("refund policy", "a.md", 0, {"team": "billing"}),
            Chunk("refund policy", "b.md", 0, {"team": "ship"}),
        ]
    )
    hits = index.search("refund policy", k=5, where={"team": "ship"})
    assert [h.chunk.source for h in hits] == ["b.md"]


def test_recall_counts_relevant_items_in_the_window() -> None:
    assert recall_at_k(["a", "b", "c"], ["b", "z"], k=3) == 0.5
    assert recall_at_k(["a", "b", "c"], ["b", "z"], k=1) == 0.0


def test_reciprocal_rank_uses_the_first_hit() -> None:
    assert reciprocal_rank(["a", "b", "c"], ["c"]) == (
        pytest.approx(1 / 3)
    )
    assert reciprocal_rank(["a"], ["z"]) == 0.0


def test_evaluate_averages_over_the_query_set() -> None:
    score = evaluate(
        results={"q1": ["a", "b"], "q2": ["z", "y"]},
        gold={"q1": ["a"], "q2": ["y"]},
        k=2,
    )
    assert score.recall_at_k == pytest.approx(1.0)
    assert score.mrr == pytest.approx((1.0 + 0.5) / 2)
    assert score.queries == 2


def test_empty_index_returns_nothing(
    embedder: OpenAIEmbedder,
) -> None:
    assert SemanticIndex(embedder).search("anything") == []
