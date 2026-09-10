import json
from pathlib import Path

import pytest

from llmapp.llm.base import Completion, Message
from llmapp.rag.ingest import ingest_directory
from llmapp.rag.rewrite import MultiQueryRetriever
from llmapp.retrieval.chunking import Chunk
from llmapp.retrieval.fusion import reciprocal_rank_fusion
from llmapp.retrieval.hybrid import HybridRetriever
from llmapp.retrieval.lexical import BM25Index, tokenize
from llmapp.retrieval.rerank import LLMReranker
from llmapp.retrieval.search import SemanticIndex
from llmapp.retrieval.store import Hit
from llmapp.retrieval.window import ParentExpander
from tests._fake_embedder import KeywordEmbedder

CORPUS = Path("tests/corpus")


def chunks() -> list[Chunk]:
    found = ingest_directory(CORPUS, max_words=40)
    return [c for group in found.values() for c in group]


def hit(chunk_id: str, score: float = 1.0) -> Hit:
    source, ordinal = chunk_id.split("#")
    return Hit(
        chunk=Chunk("text of " + chunk_id, source, int(ordinal)),
        score=score,
    )


class Replies:
    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: object,
        *,
        max_output_tokens: int = 512,
    ) -> Completion:
        self.calls.append(list(messages))  # type: ignore[arg-type]
        return Completion(
            self.texts[len(self.calls) - 1], 50, 20, "x", "stop"
        )


def test_tokenizer_keeps_identifiers() -> None:
    assert tokenize("Order 48812 was damaged.") == [
        "order",
        "48812",
        "was",
        "damaged",
    ]


def test_bm25_finds_an_exact_identifier() -> None:
    index = BM25Index()
    index.add(
        [
            Chunk("order 48812 shipped", "a.md", 0),
            Chunk("refunds take fourteen days", "b.md", 0),
            Chunk("delivery is slow this week", "c.md", 0),
            Chunk("escalation to a senior engineer", "d.md", 0),
        ]
    )
    hits = index.search("48812", k=2)
    assert hits[0].chunk.source == "a.md"


def test_bm25_is_degenerate_on_a_tiny_corpus() -> None:
    """IDF is zero when a term is in half the documents."""
    index = BM25Index()
    index.add(
        [
            Chunk("order 48812 shipped", "a.md", 0),
            Chunk("refunds take fourteen days", "b.md", 0),
        ]
    )
    assert index.search("48812", k=2) == []


def test_bm25_returns_nothing_for_unmatched_terms() -> None:
    index = BM25Index()
    index.add([Chunk("refund policy", "a.md", 0)])
    assert index.search("quarterly revenue luxembourg") == []


def test_fusion_rewards_agreement_between_rankings() -> None:
    left = [hit("a.md#0"), hit("b.md#0"), hit("c.md#0")]
    right = [hit("c.md#0"), hit("b.md#0"), hit("d.md#0")]
    fused = reciprocal_rank_fusion([left, right], k=4)
    ids = [h.chunk.chunk_id for h in fused]
    assert ids[0] in ("b.md#0", "c.md#0")
    assert set(ids) == {"a.md#0", "b.md#0", "c.md#0", "d.md#0"}


def test_fusion_ignores_incomparable_score_scales() -> None:
    big = [hit("a.md#0", 900.0), hit("b.md#0", 800.0)]
    small = [hit("b.md#0", 0.9), hit("a.md#0", 0.1)]
    fused = reciprocal_rank_fusion([big, small], k=2)
    assert {h.chunk.chunk_id for h in fused} == {
        "a.md#0",
        "b.md#0",
    }
    assert fused[0].score < 1.0


def test_a_document_found_by_only_one_arm_survives() -> None:
    semantic = [hit("a.md#0")]
    lexical = [hit("z.md#0")]
    fused = reciprocal_rank_fusion([semantic, lexical], k=2)
    assert {h.chunk.chunk_id for h in fused} == {
        "a.md#0",
        "z.md#0",
    }


def test_hybrid_combines_both_arms() -> None:
    all_chunks = chunks()
    semantic = SemanticIndex(KeywordEmbedder())
    semantic.add(all_chunks)
    lexical = BM25Index()
    lexical.add(all_chunks)
    hybrid = HybridRetriever(semantic=semantic, lexical=lexical)
    hits = hybrid.search("refund window fourteen days", k=3)
    assert hits
    assert hits[0].chunk.source == "refunds.md"


def test_reranker_reorders_by_model_score() -> None:
    client = Replies(
        json.dumps(
            [
                {"id": "a.md#0", "score": 2},
                {"id": "b.md#0", "score": 9},
            ]
        )
    )
    reranked = LLMReranker(client).rerank(
        "which policy", [hit("a.md#0"), hit("b.md#0")], k=2
    )
    assert [h.chunk.chunk_id for h in reranked] == [
        "b.md#0",
        "a.md#0",
    ]
    assert reranked[0].score == 9.0


def test_reranker_falls_back_when_output_is_unusable() -> None:
    client = Replies("I think the second one is better.")
    original = [hit("a.md#0"), hit("b.md#0")]
    reranked = LLMReranker(client).rerank("q", original, k=2)
    assert [h.chunk.chunk_id for h in reranked] == [
        "a.md#0",
        "b.md#0",
    ]


def test_reranker_sees_the_passages_not_the_answer() -> None:
    client = Replies("[]")
    LLMReranker(client).rerank("why", [hit("a.md#0")], k=1)
    system = client.calls[0][0].content
    assert "do not answer the question" in system.lower()


def test_rewriter_keeps_the_original_query() -> None:
    client = Replies("refund window\nreturn period\nfourteen days")
    retriever = MultiQueryRetriever(
        client, BM25Index(), variants=3
    )
    queries = retriever.queries("how long to send something back")
    assert queries[0] == "how long to send something back"
    assert "return period" in queries
    assert len(queries) == 4


def test_rewriter_deduplicates_variants() -> None:
    client = Replies("refund window\nrefund window")
    retriever = MultiQueryRetriever(
        client, BM25Index(), variants=2
    )
    assert len(retriever.queries("refund window")) == 1


def test_multi_query_fuses_every_variant() -> None:
    all_chunks = chunks()
    lexical = BM25Index()
    lexical.add(all_chunks)
    client = Replies("broken replacement\nrefund fourteen days")
    retriever = MultiQueryRetriever(
        client, lexical, variants=2
    )
    hits = retriever.search("something arrived smashed", k=3)
    assert any(h.chunk.source == "refunds.md" for h in hits)


def test_parent_expansion_adds_neighbouring_chunks() -> None:
    pieces = [
        Chunk("first part", "policy.md", 0),
        Chunk("the answer", "policy.md", 1),
        Chunk("third part", "policy.md", 2),
    ]
    expanded = ParentExpander(pieces, window=1).expand(
        [Hit(chunk=pieces[1], score=0.9)]
    )
    assert "first part" in expanded[0].chunk.text
    assert "third part" in expanded[0].chunk.text
    assert expanded[0].score == 0.9


def test_expansion_does_not_cross_documents() -> None:
    pieces = [
        Chunk("a tail", "a.md", 0),
        Chunk("b head", "b.md", 0),
    ]
    expanded = ParentExpander(pieces, window=2).expand(
        [Hit(chunk=pieces[1], score=0.5)]
    )
    assert "a tail" not in expanded[0].chunk.text
