import json
from pathlib import Path

import pytest

from llmapp.eval.dataset import (
    Case,
    Dataset,
    check_coverage,
    load_dataset,
)
from llmapp.eval.judge import (
    Judge,
    position_bias,
    win_rate,
)
from llmapp.eval.metrics import (
    abstention_correct,
    cohens_kappa,
    agreement,
    context_precision,
    context_recall,
    ndcg_at_k,
)
from llmapp.eval.runner import (
    compare_reports,
    evaluate_dataset,
    gate,
)
from llmapp.llm.base import Completion, Message
from llmapp.prompts.registry import PromptRegistry
from llmapp.rag.pipeline import RagPipeline, build_index_from
from llmapp.retrieval.search import SemanticIndex
from tests._fake_embedder import KeywordEmbedder

REGISTRY = PromptRegistry(Path("prompts"))
CORPUS = Path("tests/corpus")


# --- datasets -------------------------------------------------


def test_the_fingerprint_changes_with_the_cases() -> None:
    one = Dataset("d", (Case("a", "q1"),))
    two = Dataset("d", (Case("a", "q2"),))
    assert one.fingerprint != two.fingerprint
    assert len(one.fingerprint) == 8


def test_a_dataset_round_trips_through_jsonl(
    tmp_path: Path,
) -> None:
    original = Dataset(
        "d",
        (
            Case(
                "a",
                "how long for refunds",
                relevant_ids=("refunds.md#0",),
                must_contain=("fourteen",),
                tags=("policy",),
                source="production",
            ),
            Case("b", "unrelated", answerable=False),
        ),
    )
    path = tmp_path / "d.jsonl"
    original.save(path)
    loaded = load_dataset(path, "d")
    assert loaded.fingerprint == original.fingerprint
    assert loaded.unanswerable == 1


def test_a_flattering_dataset_is_flagged() -> None:
    warnings = check_coverage(
        [Case(str(i), "q", answerable=True) for i in range(5)]
    )
    assert any("only 5 cases" in w for w in warnings)
    assert any("unanswerable" in w for w in warnings)
    assert any("production" in w for w in warnings)


def test_a_healthy_dataset_raises_no_warnings() -> None:
    cases = [
        Case(
            str(i),
            "q",
            answerable=i % 4 != 0,
            source="production",
        )
        for i in range(24)
    ]
    assert check_coverage(cases) == []


def test_tag_filtering_narrows_the_set() -> None:
    data = Dataset(
        "d",
        (
            Case("a", "q", tags=("billing",)),
            Case("b", "q", tags=("shipping",)),
        ),
    )
    assert len(data.tagged("billing")) == 1


# --- metrics --------------------------------------------------


def test_ndcg_rewards_a_higher_rank() -> None:
    first = ndcg_at_k(["a", "x", "y"], ["a"], k=3)
    third = ndcg_at_k(["x", "y", "a"], ["a"], k=3)
    assert first == pytest.approx(1.0)
    assert third < first
    assert third > 0.0


def test_ndcg_is_zero_when_nothing_relevant_is_found() -> None:
    assert ndcg_at_k(["x", "y"], ["a"], k=2) == 0.0


def test_recall_and_precision_answer_different_questions(
) -> None:
    retrieved = ["a", "x", "y", "z"]
    relevant = ["a"]
    assert context_recall(retrieved, relevant) == 1.0
    assert context_precision(retrieved, relevant) == 0.25


def test_abstention_is_correct_in_both_directions() -> None:
    assert abstention_correct(True, True)
    assert abstention_correct(False, False)
    assert not abstention_correct(True, False)
    assert not abstention_correct(False, True)


def test_kappa_discounts_agreement_by_chance() -> None:
    left = ["good"] * 9 + ["bad"]
    right = ["good"] * 9 + ["bad"]
    assert agreement(left, right) == 1.0
    assert cohens_kappa(left, right) == pytest.approx(1.0)

    lazy = ["good"] * 10
    honest = ["good"] * 9 + ["bad"]
    assert agreement(lazy, honest) == pytest.approx(0.9)
    # High raw agreement, no skill.
    assert cohens_kappa(lazy, honest) == pytest.approx(0.0)


# --- the judge ------------------------------------------------


class FixedJudge:
    """A judge with a deliberate flaw, for calibration tests."""

    def __init__(self, always: str) -> None:
        self.always = always
        self.calls = 0

    def complete(
        self,
        messages: object,
        *,
        max_output_tokens: int = 512,
    ) -> Completion:
        self.calls += 1
        payload = json.dumps(
            {"winner": self.always, "reason": "position"}
        )
        return Completion(payload, 50, 20, "stub", "stop")


def test_position_bias_is_revealed_by_swapping() -> None:
    """A judge that always picks the first answer."""
    judge = Judge(FixedJudge("A"), REGISTRY)
    winner, agreed = judge.compare_both_ways("q", "left", "right")
    assert agreed is False
    assert winner == "tie"
    assert judge.client.calls == 2  # type: ignore[attr-defined]


def test_a_consistent_judge_agrees_with_itself() -> None:
    class PrefersLeftText:
        def complete(
            self,
            messages: object,
            *,
            max_output_tokens: int = 512,
        ) -> Completion:
            body = messages[-1].content  # type: ignore[index]
            first = body.find("better answer")
            second = body.find("worse answer")
            winner = "A" if 0 <= first < second else "B"
            return Completion(
                json.dumps(
                    {"winner": winner, "reason": "content"}
                ),
                50,
                20,
                "stub",
                "stop",
            )

    judge = Judge(PrefersLeftText(), REGISTRY)
    winner, agreed = judge.compare_both_ways(
        "q", "better answer", "worse answer"
    )
    assert agreed is True
    assert winner == "A"


def test_position_bias_is_a_measured_rate() -> None:
    pairs = [("A", True), ("B", False), ("A", False), ("A", True)]
    assert position_bias(pairs) == 0.5


def test_win_rate_ignores_ties() -> None:
    assert win_rate(["A", "B", "tie", "B"]) == pytest.approx(2 / 3)


# --- the harness ----------------------------------------------


class Answers:
    def __init__(self, payloads: dict[str, object]) -> None:
        self.payloads = payloads

    def complete(
        self,
        messages: object,
        *,
        max_output_tokens: int = 512,
    ) -> Completion:
        body = messages[-1].content  # type: ignore[index]
        for marker, payload in self.payloads.items():
            if marker in body:
                return Completion(
                    json.dumps(payload), 90, 30, "stub", "stop"
                )
        return Completion(
            json.dumps(
                {"answered": False, "missing": "no match"}
            ),
            90,
            30,
            "stub",
            "stop",
        )


def build_pipeline(payloads: dict[str, object]) -> RagPipeline:
    index = SemanticIndex(KeywordEmbedder())
    build_index_from(CORPUS, index, max_words=60)
    return RagPipeline(Answers(payloads), index, REGISTRY)


DATASET = Dataset(
    "triage",
    (
        Case(
            "refund-window",
            "refunds fourteen days",
            relevant_ids=("refunds.md#0",),
            must_contain=("fourteen",),
            source="production",
        ),
        Case(
            "off-corpus",
            "quarterly revenue luxembourg",
            answerable=False,
            source="production",
        ),
    ),
)


def test_the_harness_scores_a_dataset() -> None:
    pipeline = build_pipeline(
        {
            "refunds": {
                "answered": True,
                "answer": (
                    "Refunds are issued within fourteen days."
                ),
                "citations": [
                    {
                        "chunk_id": "refunds.md#0",
                        "quote": "fourteen days",
                    }
                ],
                "missing": "",
            }
        }
    )
    report = evaluate_dataset(
        DATASET, lambda case: pipeline.answer(case.question)
    )
    assert report.pass_rate == 1.0
    assert report.abstention_rate == 1.0
    assert report.mean_recall == 1.0
    assert report.failures() == []
    assert report.fingerprint == DATASET.fingerprint


def test_a_hallucinated_answer_fails_the_case() -> None:
    pipeline = build_pipeline(
        {
            "refunds": {
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
        }
    )
    report = evaluate_dataset(
        DATASET, lambda case: pipeline.answer(case.question)
    )
    assert "refund-window" in report.failures()
    assert report.pass_rate == 0.5


def test_the_gate_blocks_a_regression() -> None:
    good = build_pipeline(
        {
            "refunds": {
                "answered": True,
                "answer": (
                    "Refunds are issued within fourteen days."
                ),
                "citations": [
                    {
                        "chunk_id": "refunds.md#0",
                        "quote": "fourteen days",
                    }
                ],
                "missing": "",
            }
        }
    )
    baseline = evaluate_dataset(
        DATASET, lambda c: good.answer(c.question)
    )
    bad = build_pipeline({})
    after = evaluate_dataset(
        DATASET, lambda c: bad.answer(c.question)
    )
    passed, reason = gate(
        after, min_pass_rate=0.9, baseline=baseline
    )
    assert passed is False
    changes = compare_reports(baseline, after)
    assert any("REGRESSED" in line for line in changes)


def test_runs_on_different_datasets_are_not_comparable() -> None:
    pipeline = build_pipeline({})
    first = evaluate_dataset(
        DATASET, lambda c: pipeline.answer(c.question)
    )
    other = Dataset("triage", DATASET.cases[:1])
    second = evaluate_dataset(
        other, lambda c: pipeline.answer(c.question)
    )
    assert compare_reports(first, second) == [
        "dataset changed; runs are not comparable"
    ]


def test_a_report_saves_a_readable_record(tmp_path: Path) -> None:
    pipeline = build_pipeline({})
    report = evaluate_dataset(
        DATASET, lambda c: pipeline.answer(c.question)
    )
    path = tmp_path / "report.json"
    report.save(path)
    saved = json.loads(path.read_text())
    assert saved["dataset"] == "triage"
    assert "pass_rate" in saved
