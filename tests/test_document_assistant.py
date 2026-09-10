"""Project 1, tested as a deliverable rather than a demo."""

import json
from pathlib import Path

import pytest

from llmapp.eval.dataset import check_coverage, load_dataset
from llmapp.eval.runner import evaluate_dataset
from llmapp.llm.base import Completion, Message
from llmapp.prompts.registry import PromptRegistry
from llmapp.projects.assistant import (
    DocumentAssistant,
    ingest_corpus,
    meta_for,
)
from llmapp.rag.access import AccessDenied, Principal
from llmapp.retrieval.local import HashEmbedder

CORPUS = Path("corpus")
REGISTRY = PromptRegistry(Path("prompts"))
BILLING = Principal(user_id="u1", team="billing")
SHIPPING = Principal(user_id="u2", team="shipping")


class Answers:
    """Canned JSON keyed on a marker in the prompt."""

    def __init__(self, payloads: dict[str, object]) -> None:
        self.payloads = payloads
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: object,
        *,
        max_output_tokens: int = 512,
    ) -> Completion:
        self.calls.append(list(messages))  # type: ignore[arg-type]
        body = self.calls[-1][-1].content
        for marker, payload in self.payloads.items():
            if marker in body:
                return Completion(
                    json.dumps(payload), 900, 60, "stub", "stop"
                )
        return Completion(
            json.dumps(
                {"answered": False, "missing": "no passage"}
            ),
            900,
            30,
            "stub",
            "stop",
        )


def assistant(payloads: dict[str, object]) -> DocumentAssistant:
    built = DocumentAssistant(
        client=Answers(payloads),
        embedder=HashEmbedder(256),
        registry=REGISTRY,
    )
    built.load(CORPUS)
    return built


def cited(answer: str, chunk_id: str, quote: str) -> dict[str, object]:
    return {
        "answered": True,
        "answer": answer,
        "citations": [{"chunk_id": chunk_id, "quote": quote}],
        "missing": "",
    }


# Scripted replies, one per answerable case. The wording is
# authored; the pipeline, retrieval, and checks are real.
GOOD: dict[str, object] = {
    "fourteen days": cited(
        "Refunds are issued within fourteen days.",
        "refunds-2026.md#0",
        "fourteen days",
    ),
    "photograph": cited(
        "Broken items qualify for a replacement at no cost.",
        "refunds-2026.md#1",
        "replacement at no cost",
    ),
    "declined without exception": cited(
        "Claims after ninety days are declined.",
        "refunds-2026.md#2",
        "declined without exception",
    ),
    "refunded\nseparately": cited(
        "Surcharges are refunded separately from the item.",
        "delivery.md#0",
        "refunded",
    ),
    "senior engineer": cited(
        "Unresolved tickets go to a senior engineer.",
        "escalation.md#0",
        "senior engineer",
    ),
}


# --- ingestion and labeling -----------------------------------


def test_labels_come_from_the_directory_not_the_text() -> None:
    meta = meta_for(
        CORPUS / "billing" / "refunds-2026.md", CORPUS
    )
    assert meta.team == "billing"
    assert meta.effective_date == "2026-01-01"


def test_the_superseded_document_carries_an_older_date() -> None:
    meta = meta_for(
        CORPUS / "billing" / "refunds-2024.md", CORPUS
    )
    assert meta.effective_date == "2024-01-01"


def test_every_chunk_is_labeled_and_traceable() -> None:
    chunks = ingest_corpus(CORPUS)
    assert len(chunks) >= 8
    for chunk in chunks:
        assert chunk.metadata["team"] in {
            "billing",
            "shipping",
            "support",
        }
        assert chunk.metadata["effective_date"]
        assert chunk.chunk_id.count("#") == 1


def test_both_indexes_receive_the_same_chunks() -> None:
    built = assistant({})
    assert len(built.semantic) == len(built.lexical)
    assert len(built) >= 8


# --- retrieval under permissions ------------------------------


def test_two_teams_see_different_documents() -> None:
    built = assistant({})
    billing = built.retrieve("refunds fourteen days", BILLING)
    shipping = built.retrieve("refunds fourteen days", SHIPPING)
    assert {h.chunk.metadata["team"] for h in billing} == {
        "billing"
    }
    assert {h.chunk.metadata["team"] for h in shipping} <= {
        "shipping"
    }


def test_hybrid_retrieval_finds_an_exact_phrase() -> None:
    built = assistant({})
    hits = built.retrieve("senior engineer", Principal("u3", "support"))
    assert any(
        "escalation.md" in h.chunk.source for h in hits
    )


def test_a_caller_cannot_widen_their_team_filter() -> None:
    built = assistant({})
    with pytest.raises(AccessDenied, match="team"):
        built.retrieve(
            "delivery", BILLING, where={"team": "shipping"}
        )


def test_a_date_filter_excludes_the_superseded_policy() -> None:
    built = assistant({})
    hits = built.retrieve(
        "refunds window",
        BILLING,
        where={"effective_date": "2026-01-01"},
    )
    assert all(
        "2024" not in h.chunk.source for h in hits
    )


# --- answering ------------------------------------------------


def test_a_grounded_answer_is_returned_with_citations() -> None:
    built = assistant(GOOD)
    result = built.answer("refunds fourteen days", BILLING)
    assert result.answer.answered is True
    assert result.answer.citations[0].chunk_id.startswith(
        "refunds-2026.md"
    )
    assert result.grounding is not None
    assert result.grounding.is_grounded


def test_an_off_corpus_question_is_refused_without_a_call(
) -> None:
    built = assistant(GOOD)
    result = built.answer(
        "quarterly revenue luxembourg subsidiary", BILLING
    )
    assert result.answer.answered is False
    assert built.client.calls == []  # type: ignore[attr-defined]


def test_a_substituted_number_is_caught() -> None:
    built = assistant(
        {
            "fourteen days": {
                "answered": True,
                "answer": "Refunds are issued within ninety days.",
                "citations": [
                    {
                        "chunk_id": "refunds-2026.md#0",
                        "quote": "Refunds are issued within",
                    }
                ],
                "missing": "",
            }
        }
    )
    result = built.answer("refunds fourteen days", BILLING)
    assert result.answer.answered is False


# --- the evaluation dataset -----------------------------------


def test_the_project_dataset_is_not_flattering() -> None:
    dataset = load_dataset(
        Path("evals/assistant.jsonl"), "assistant"
    )
    warnings = check_coverage(list(dataset))
    assert dataset.unanswerable >= 2
    assert not any("unanswerable" in w for w in warnings)
    assert not any("production" in w for w in warnings)


def test_the_harness_runs_against_the_project() -> None:
    built = assistant(GOOD)
    dataset = load_dataset(
        Path("evals/assistant.jsonl"), "assistant"
    )
    report = evaluate_dataset(
        dataset, lambda case: built.answer(case.question, BILLING)
    )
    assert report.fingerprint == dataset.fingerprint
    assert 0.0 <= report.pass_rate <= 1.0
    assert report.abstention_rate > 0.0
