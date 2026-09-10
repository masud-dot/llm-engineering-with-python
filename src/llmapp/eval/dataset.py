"""Golden datasets: the thing evaluation is measured against."""

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Case:
    """One labeled example."""

    id: str
    question: str
    expected: str = ""
    relevant_ids: tuple[str, ...] = ()
    must_contain: tuple[str, ...] = ()
    must_not_contain: tuple[str, ...] = ()
    answerable: bool = True
    tags: tuple[str, ...] = ()
    source: str = "authored"

    def to_json(self) -> dict[str, object]:
        return {
            "id": self.id,
            "question": self.question,
            "expected": self.expected,
            "relevant_ids": list(self.relevant_ids),
            "must_contain": list(self.must_contain),
            "must_not_contain": list(self.must_not_contain),
            "answerable": self.answerable,
            "tags": list(self.tags),
            "source": self.source,
        }


@dataclass(frozen=True)
class Dataset:
    """A versioned set of cases."""

    name: str
    cases: tuple[Case, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[Case]:
        return iter(self.cases)

    @property
    def fingerprint(self) -> str:
        """Changes whenever any case changes."""
        blob = json.dumps(
            [case.to_json() for case in self.cases],
            sort_keys=True,
        ).encode()
        return hashlib.sha256(blob).hexdigest()[:8]

    def tagged(self, tag: str) -> "Dataset":
        return Dataset(
            name=f"{self.name}[{tag}]",
            cases=tuple(c for c in self.cases if tag in c.tags),
        )

    @property
    def unanswerable(self) -> int:
        return sum(1 for c in self.cases if not c.answerable)

    def save(self, path: Path) -> None:
        lines = [
            json.dumps(case.to_json(), sort_keys=True)
            for case in self.cases
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_dataset(path: Path, name: str = "") -> Dataset:
    """Read a JSONL dataset from disk."""
    cases: list[Case] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        cases.append(
            Case(
                id=raw["id"],
                question=raw["question"],
                expected=raw.get("expected", ""),
                relevant_ids=tuple(raw.get("relevant_ids", [])),
                must_contain=tuple(raw.get("must_contain", [])),
                must_not_contain=tuple(
                    raw.get("must_not_contain", [])
                ),
                answerable=raw.get("answerable", True),
                tags=tuple(raw.get("tags", [])),
                source=raw.get("source", "authored"),
            )
        )
    return Dataset(name=name or path.stem, cases=tuple(cases))


def check_coverage(dataset: Sequence[Case]) -> list[str]:
    """Warn about a dataset that will report a flattering number."""
    warnings: list[str] = []
    if len(dataset) < 20:
        warnings.append(
            f"only {len(dataset)} cases; too few to detect a "
            "small regression"
        )
    unanswerable = sum(1 for c in dataset if not c.answerable)
    if unanswerable == 0:
        warnings.append(
            "no unanswerable cases; over-answering is invisible"
        )
    from_production = sum(
        1 for c in dataset if c.source == "production"
    )
    if from_production == 0:
        warnings.append(
            "no cases from production; the set may test only "
            "vocabulary the documents already use"
        )
    return warnings
