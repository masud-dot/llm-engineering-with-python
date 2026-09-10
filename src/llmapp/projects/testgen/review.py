"""Score a generated suite before a human reads it."""

import ast
from collections.abc import Sequence
from dataclasses import dataclass

from llmapp.projects.testgen.cases import TestCase
from llmapp.projects.testgen.plan import CaseKind, Slot

TRIVIAL_STATUSES = frozenset({200, 201, 204})


@dataclass(frozen=True)
class Review:
    """What a reviewer would otherwise check by hand."""

    planned: int
    generated: int
    covered_kinds: frozenset[str]
    missing_kinds: frozenset[str]
    duplicates: tuple[str, ...]
    trivial: tuple[str, ...]
    parses: bool

    @property
    def coverage(self) -> float:
        total = len(self.covered_kinds | self.missing_kinds)
        return len(self.covered_kinds) / total if total else 0.0

    @property
    def duplicate_rate(self) -> float:
        return (
            len(self.duplicates) / self.generated
            if self.generated
            else 0.0
        )

    @property
    def accepted(self) -> bool:
        return (
            self.parses
            and not self.missing_kinds
            and not self.duplicates
        )

    def summary(self) -> str:
        return (
            f"{self.generated}/{self.planned} cases  "
            f"coverage {self.coverage:.0%}  "
            f"duplicates {len(self.duplicates)}  "
            f"trivial {len(self.trivial)}  "
            f"parses {self.parses}"
        )


def signature(case: TestCase) -> str:
    """Two cases doing the same thing, whatever they are named."""
    body = sorted(
        f"{k}={v!r}" for k, v in sorted(case.body.items())
    )
    return "|".join(
        [
            case.method,
            case.path,
            str(case.expected_status),
            ",".join(sorted(case.headers)),
            ",".join(body),
        ]
    )


def review(
    slots: Sequence[Slot],
    cases: Sequence[TestCase],
    source: str,
) -> Review:
    planned_kinds = {slot.kind.value for slot in slots}
    produced_kinds = {case.kind.value for case in cases}

    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for case in cases:
        key = signature(case)
        if key in seen:
            duplicates.append(case.name)
        else:
            seen[key] = case.name

    trivial = [
        case.name
        for case in cases
        if case.kind is not CaseKind.HAPPY
        and case.expected_status in TRIVIAL_STATUSES
    ]

    try:
        ast.parse(source)
        parses = True
    except SyntaxError:
        parses = False

    return Review(
        planned=len(slots),
        generated=len(cases),
        covered_kinds=frozenset(produced_kinds & planned_kinds),
        missing_kinds=frozenset(planned_kinds - produced_kinds),
        duplicates=tuple(duplicates),
        trivial=tuple(trivial),
        parses=parses,
    )
