"""Baselines, prompt regression, and model qualification."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from llmapp.eval.dataset import Case, Dataset
from llmapp.eval.runner import (
    EvalReport,
    Runner,
    evaluate_dataset,
)

BASELINES = Path("evals/baselines")


@dataclass(frozen=True)
class Baseline:
    """A stored result to compare future runs against."""

    label: str
    fingerprint: str
    pass_rate: float
    grounded_rate: float
    mean_recall: float
    passing: frozenset[str]

    def to_json(self) -> dict[str, object]:
        return {
            "label": self.label,
            "fingerprint": self.fingerprint,
            "pass_rate": self.pass_rate,
            "grounded_rate": self.grounded_rate,
            "mean_recall": self.mean_recall,
            "passing": sorted(self.passing),
        }


def snapshot(report: EvalReport, label: str) -> Baseline:
    """Freeze a report into a comparable baseline."""
    return Baseline(
        label=label,
        fingerprint=report.fingerprint,
        pass_rate=report.pass_rate,
        grounded_rate=report.grounded_rate,
        mean_recall=report.mean_recall,
        passing=frozenset(
            s.case_id for s in report.scores if s.passed
        ),
    )


def save_baseline(baseline: Baseline, root: Path = BASELINES) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{baseline.label}.json"
    path.write_text(
        json.dumps(baseline.to_json(), indent=1), encoding="utf-8"
    )
    return path


def load_baseline(label: str, root: Path = BASELINES) -> Baseline:
    raw = json.loads(
        (root / f"{label}.json").read_text(encoding="utf-8")
    )
    return Baseline(
        label=raw["label"],
        fingerprint=raw["fingerprint"],
        pass_rate=float(raw["pass_rate"]),
        grounded_rate=float(raw["grounded_rate"]),
        mean_recall=float(raw["mean_recall"]),
        passing=frozenset(raw["passing"]),
    )


@dataclass(frozen=True)
class Regression:
    """The difference between a baseline and a new run."""

    broken: tuple[str, ...]
    fixed: tuple[str, ...]
    delta_pass_rate: float
    comparable: bool
    note: str = ""

    @property
    def is_regression(self) -> bool:
        return bool(self.broken)


def compare_to_baseline(
    baseline: Baseline, report: EvalReport
) -> Regression:
    """Per-case comparison, not just an average."""
    if baseline.fingerprint != report.fingerprint:
        return Regression(
            broken=(),
            fixed=(),
            delta_pass_rate=0.0,
            comparable=False,
            note=(
                f"dataset moved from {baseline.fingerprint} to "
                f"{report.fingerprint}"
            ),
        )
    now_passing = {s.case_id for s in report.scores if s.passed}
    return Regression(
        broken=tuple(sorted(baseline.passing - now_passing)),
        fixed=tuple(sorted(now_passing - baseline.passing)),
        delta_pass_rate=report.pass_rate - baseline.pass_rate,
        comparable=True,
    )


@dataclass(frozen=True)
class Qualification:
    """The result of trying a candidate configuration."""

    baseline_label: str
    candidate_label: str
    regression: Regression
    candidate: EvalReport

    def verdict(self, *, allow_broken: int = 0) -> tuple[bool, str]:
        if not self.regression.comparable:
            return False, self.regression.note
        broken = self.regression.broken
        if len(broken) > allow_broken:
            return False, (
                f"{len(broken)} case(s) regressed: "
                f"{', '.join(broken)}"
            )
        return True, (
            f"pass rate {self.regression.delta_pass_rate:+.0%}, "
            f"{len(self.regression.fixed)} fixed, "
            f"{len(broken)} broken"
        )


def qualify(
    dataset: Dataset,
    baseline: Baseline,
    candidate_runner: Runner,
    candidate_label: str,
) -> Qualification:
    """Run a candidate against the same dataset and compare."""
    report = evaluate_dataset(dataset, candidate_runner)
    return Qualification(
        baseline_label=baseline.label,
        candidate_label=candidate_label,
        regression=compare_to_baseline(baseline, report),
        candidate=report,
    )


def prompt_regression(
    dataset: Dataset,
    build_runner: Callable[[int], Runner],
    versions: tuple[int, ...],
) -> dict[int, EvalReport]:
    """Score several prompt versions on one dataset."""
    return {
        version: evaluate_dataset(dataset, build_runner(version))
        for version in versions
    }


def failed_cases(report: EvalReport) -> list[Case]:
    """Failures, ready to become regression cases."""
    return [
        Case(
            id=f"regression-{score.case_id}",
            question="",
            tags=("regression",),
            source="production",
        )
        for score in report.scores
        if not score.passed
    ]
