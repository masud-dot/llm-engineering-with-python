import json
from pathlib import Path

import pytest

from llmapp.llm.base import Completion, Message
from llmapp.prompts.registry import PromptRegistry
from llmapp.rag.access import AccessDenied, AccessPolicy, Principal
from llmapp.rag.answer import Citation, GroundedAnswer
from llmapp.rag.grounding import check, detect_conflicts, locate
from llmapp.rag.pipeline import RagPipeline
from llmapp.retrieval.chunking import Chunk
from llmapp.retrieval.search import SemanticIndex
from llmapp.retrieval.store import Hit
from tests._fake_embedder import KeywordEmbedder

REGISTRY = PromptRegistry(Path("prompts"))

BILLING = Chunk(
    "Refunds are issued within fourteen days of the return.",
    "refunds.md",
    0,
    {"team": "billing", "topic": "refunds",
     "effective_date": "2026-01-01"},
)
BILLING_OLD = Chunk(
    "Refunds are issued within ninety days of the return.",
    "refunds-2024.md",
    0,
    {"team": "billing", "topic": "refunds",
     "effective_date": "2024-01-01"},
)
SHIPPING = Chunk(
    "Express delivery surcharges are refunded separately.",
    "delivery.md",
    0,
    {"team": "shipping", "topic": "delivery",
     "effective_date": "2026-01-01"},
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


def answer_payload(
    chunk_id: str, quote: str, text: str
) -> dict[str, object]:
    return {
        "answered": True,
        "answer": text,
        "citations": [{"chunk_id": chunk_id, "quote": quote}],
        "missing": "",
    }


# --- grounding ------------------------------------------------


def test_a_quote_is_located_with_offsets() -> None:
    span = locate("fourteen days", BILLING.text, "refunds.md#0")
    assert span is not None
    assert BILLING.text[span.start : span.end] == "fourteen days"


def test_a_missing_quote_has_no_span() -> None:
    assert locate("ninety days", BILLING.text, "refunds.md#0") is (
        None
    )


def test_an_unsupported_sentence_is_flagged() -> None:
    answer = GroundedAnswer(
        answered=True,
        answer=(
            "Refunds are issued within fourteen days. "
            "Business customers receive priority handling."
        ),
        citations=[
            Citation(
                chunk_id="refunds.md#0", quote="fourteen days"
            )
        ],
    )
    report = check(answer, [Hit(chunk=BILLING, score=0.9)])
    assert report.spans
    assert len(report.unsupported) == 1
    assert "priority handling" in report.unsupported[0]
    assert report.coverage == pytest.approx(0.5)
    assert report.is_grounded is False


def test_a_fully_supported_answer_is_grounded() -> None:
    answer = GroundedAnswer(
        answered=True,
        answer="Refunds are issued within fourteen days.",
        citations=[
            Citation(
                chunk_id="refunds.md#0", quote="fourteen days"
            )
        ],
    )
    report = check(answer, [Hit(chunk=BILLING, score=0.9)])
    assert report.is_grounded is True
    assert report.coverage == pytest.approx(1.0)


def test_conflicting_dates_on_one_topic_are_detected() -> None:
    hits = [
        Hit(chunk=BILLING, score=0.9),
        Hit(chunk=BILLING_OLD, score=0.8),
    ]
    assert detect_conflicts(hits) == ["refunds"]


def test_agreeing_passages_raise_no_conflict() -> None:
    hits = [
        Hit(chunk=BILLING, score=0.9),
        Hit(chunk=SHIPPING, score=0.7),
    ]
    assert detect_conflicts(hits) == []


# --- the pipeline gate ----------------------------------------


def test_a_partly_unsupported_answer_is_demoted() -> None:
    client = Replies(
        answer_payload(
            "refunds.md#0",
            "fourteen days",
            "Refunds take fourteen days. Shipping is free.",
        )
    )
    pipeline = RagPipeline(client, index_of(BILLING), REGISTRY)
    result = pipeline.answer("refunds fourteen days")
    assert result.answer.answered is False
    assert "not supported" in result.answer.missing
    assert result.grounding is not None
    assert result.grounding.coverage < 1.0


def test_a_supported_answer_keeps_its_spans() -> None:
    client = Replies(
        answer_payload(
            "refunds.md#0",
            "fourteen days",
            "Refunds are issued within fourteen days.",
        )
    )
    pipeline = RagPipeline(client, index_of(BILLING), REGISTRY)
    result = pipeline.answer("refunds fourteen days")
    assert result.answer.answered is True
    assert result.grounding is not None
    span = result.grounding.spans[0]
    assert BILLING.text[span.start : span.end] == "fourteen days"


# --- access control -------------------------------------------


def test_two_users_get_different_evidence() -> None:
    index = index_of(BILLING, SHIPPING)
    policy = AccessPolicy()
    client = Replies(
        {"answered": False, "missing": "n/a"},
        {"answered": False, "missing": "n/a"},
    )
    pipeline = RagPipeline(
        client, index, REGISTRY, policy=policy, min_score=0.0
    )
    billing = pipeline.answer(
        "refunds delivery",
        principal=Principal("u1", team="billing"),
    )
    shipping = pipeline.answer(
        "refunds delivery",
        principal=Principal("u2", team="shipping"),
    )
    assert billing.sources == ["refunds.md"]
    assert shipping.sources == ["delivery.md"]


def test_a_pipeline_with_a_policy_requires_a_principal() -> None:
    pipeline = RagPipeline(
        Replies(), index_of(BILLING), REGISTRY,
        policy=AccessPolicy(),
    )
    with pytest.raises(PermissionError, match="principal"):
        pipeline.answer("refunds")


def test_a_caller_cannot_widen_their_own_filter() -> None:
    pipeline = RagPipeline(
        Replies(), index_of(BILLING, SHIPPING), REGISTRY,
        policy=AccessPolicy(), min_score=0.0,
    )
    with pytest.raises(AccessDenied, match="team"):
        pipeline.answer(
            "delivery",
            where={"team": "shipping"},
            principal=Principal("u1", team="billing"),
        )


def test_a_caller_may_narrow_within_their_permissions() -> None:
    client = Replies({"answered": False, "missing": "n/a"})
    pipeline = RagPipeline(
        client, index_of(BILLING, SHIPPING), REGISTRY,
        policy=AccessPolicy(), min_score=0.0,
    )
    result = pipeline.answer(
        "refunds",
        where={"topic": "refunds"},
        principal=Principal("u1", team="billing"),
    )
    assert result.sources == ["refunds.md"]


def test_a_substituted_number_is_caught_by_the_quantity_check(
) -> None:
    """Word overlap alone would pass this; quantities catch it."""
    from llmapp.rag.grounding import quantities

    answer = GroundedAnswer(
        answered=True,
        answer="Refunds are issued within ninety days.",
        citations=[
            Citation(
                chunk_id="refunds.md#0",
                quote="Refunds are issued within",
            )
        ],
    )
    report = check(answer, [Hit(chunk=BILLING, score=0.9)])
    assert report.unsupported == [
        "Refunds are issued within ninety days."
    ]
    assert quantities("ninety days") == {"ninety"}
    assert "fourteen" in quantities(BILLING.text)


def test_digits_are_checked_as_well_as_number_words() -> None:
    answer = GroundedAnswer(
        answered=True,
        answer="Refunds are issued within 90 days.",
        citations=[
            Citation(
                chunk_id="refunds.md#0",
                quote="Refunds are issued within",
            )
        ],
    )
    report = check(answer, [Hit(chunk=BILLING, score=0.9)])
    assert report.unsupported
