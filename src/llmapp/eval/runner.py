"""Run a dataset through a pipeline and score the results."""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from llmapp.eval.dataset import Case, Dataset
from llmapp.eval.metrics import (
    abstention_correct,
    contains_all,
    contains_none,
    context_precision,
    context_recall,
    ndcg_at_k,
)
from llmapp.rag.pipeline import RagResult

Runner = Callable[[Case], RagResult]


@dataclass(frozen=True)
class CaseScore:
    """Every measurement for one case."""

    case_id: str
    answered: bool
    abstention_ok: bool
    recall: float | None
    precision: float | None
    ndcg: float | None
    facts_ok: bool
    forbidden_ok: bool
    grounded: bool
    coverage: float
    repairs: int

    @property
    def passed(self) -> bool:
        return (
            self.abstention_ok
            and self.facts_ok
            and self.forbidden_ok
            and (self.grounded or not self.answered)
        )


@dataclass(frozen=True)
class EvalReport:
    """The result of one evaluation run."""

    dataset: str
    fingerprint: str
    prompt_ids: tuple[str, ...]
    scores: tuple[CaseScore, ...]
    at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    def _mean(self, values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @property
    def pass_rate(self) -> float:
        return self._mean(
            [1.0 if s.passed else 0.0 for s in self.scores]
        )

    @property
    def abstention_rate(self) -> float:
        return self._mean(
            [1.0 if s.abstention_ok else 0.0 for s in self.scores]
        )

    @property
    def mean_recall(self) -> float:
        return self._mean(
            [s.recall for s in self.scores if s.recall is not None]
        )

    @property
    def mean_precision(self) -> float:
        return self._mean(
            [
                s.precision
                for s in self.scores
                if s.precision is not None
            ]
        )

    @property
    def mean_ndcg(self) -> float:
        return self._mean(
            [s.ndcg for s in self.scores if s.ndcg is not None]
        )

    @property
    def grounded_rate(self) -> float:
        answered = [s for s in self.scores if s.answered]
        if not answered:
            return 0.0
        return self._mean(
            [1.0 if s.grounded else 0.0 for s in answered]
        )

    @property
    def repair_rate(self) -> float:
        return self._mean(
            [1.0 if s.repairs > 1 else 0.0 for s in self.scores]
        )

    def failures(self) -> list[str]:
        return [s.case_id for s in self.scores if not s.passed]

    def summary(self) -> str:
        return (
            f"{self.dataset}@{self.fingerprint}  "
            f"pass {self.pass_rate:.0%}  "
            f"recall {self.mean_recall:.2f}  "
            f"ndcg {self.mean_ndcg:.2f}  "
            f"precision {self.mean_precision:.2f}  "
            f"grounded {self.grounded_rate:.0%}  "
            f"abstention {self.abstention_rate:.0%}"
        )

    def save(self, path: Path) -> None:
        payload = {
            "dataset": self.dataset,
            "fingerprint": self.fingerprint,
            "at": self.at,
            "prompt_ids": list(self.prompt_ids),
            "pass_rate": self.pass_rate,
            "mean_recall": self.mean_recall,
            "mean_ndcg": self.mean_ndcg,
            "grounded_rate": self.grounded_rate,
            "failures": self.failures(),
        }
        path.write_text(
            json.dumps(payload, indent=1), encoding="utf-8"
        )


def score_case(case: Case, result: RagResult, k: int = 5) -> CaseScore:
    """Turn one pipeline result into measurements."""
    retrieved = [h.chunk.chunk_id for h in result.hits]
    answered = result.answer.answered
    grounding = result.grounding
    return CaseScore(
        case_id=case.id,
        answered=answered,
        abstention_ok=abstention_correct(answered, case.answerable),
        recall=(
            context_recall(retrieved, case.relevant_ids)
            if case.relevant_ids
            else None
        ),
        precision=(
            context_precision(retrieved, case.relevant_ids)
            if case.relevant_ids
            else None
        ),
        ndcg=(
            ndcg_at_k(retrieved, case.relevant_ids, k)
            if case.relevant_ids
            else None
        ),
        facts_ok=contains_all(
            result.answer.answer, case.must_contain
        ),
        forbidden_ok=contains_none(
            result.answer.answer, case.must_not_contain
        ),
        grounded=bool(grounding and grounding.is_grounded),
        coverage=grounding.coverage if grounding else 0.0,
        repairs=result.report.attempts,
    )


def evaluate_dataset(
    dataset: Dataset, run: Runner, k: int = 5
) -> EvalReport:
    """Run every case and collect the measurements."""
    scores: list[CaseScore] = []
    prompt_ids: set[str] = set()
    for case in dataset:
        result = run(case)
        prompt_ids.add(result.prompt_id)
        scores.append(score_case(case, result, k))
    return EvalReport(
        dataset=dataset.name,
        fingerprint=dataset.fingerprint,
        prompt_ids=tuple(sorted(p for p in prompt_ids if p)),
        scores=tuple(scores),
    )


def compare_reports(
    before: EvalReport, after: EvalReport
) -> list[str]:
    """What changed, per case, between two runs."""
    if before.fingerprint != after.fingerprint:
        return ["dataset changed; runs are not comparable"]
    old = {s.case_id: s.passed for s in before.scores}
    lines: list[str] = []
    for score in after.scores:
        was = old.get(score.case_id)
        if was is None or was == score.passed:
            continue
        lines.append(
            f"{score.case_id}: "
            f"{'fixed' if score.passed else 'REGRESSED'}"
        )
    return lines


def gate(
    report: EvalReport,
    *,
    min_pass_rate: float,
    baseline: EvalReport | None = None,
) -> tuple[bool, str]:
    """The decision a pipeline can act on."""
    if report.pass_rate < min_pass_rate:
        return False, (
            f"pass rate {report.pass_rate:.0%} is below the "
            f"floor of {min_pass_rate:.0%}"
        )
    if baseline is not None:
        changes = compare_reports(baseline, report)
        regressions = [c for c in changes if "REGRESSED" in c]
        if regressions:
            return False, "; ".join(regressions)
    return True, "ok"
