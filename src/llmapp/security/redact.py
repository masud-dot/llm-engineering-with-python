"""Remove sensitive strings before they are stored or sent.

Pattern redaction is a floor, not a boundary. It catches
shapes — keys, cards, emails — and cannot catch a customer's
name. The stronger position is to store an identifier and
keep the content in a system with the same access controls
as its source.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

PLACEHOLDER = "[redacted:{label}]"


@dataclass(frozen=True)
class Pattern:
    """One shape worth removing, with a label for the log."""

    label: str
    regex: re.Pattern[str]


DEFAULT_PATTERNS: tuple[Pattern, ...] = (
    Pattern("openai_key", re.compile(r"sk-[A-Za-z0-9_-]{20,}")),
    Pattern("aws_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    Pattern(
        "bearer",
        re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{20,}"),
    ),
    Pattern(
        "email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}")
    ),
    Pattern("card", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    Pattern(
        "iban",
        re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
    ),
)


@dataclass
class Redactor:
    """Redact text and report what was found."""

    patterns: tuple[Pattern, ...] = DEFAULT_PATTERNS
    found: dict[str, int] = field(default_factory=dict)

    def scrub(self, text: str) -> str:
        cleaned = text
        for pattern in self.patterns:
            cleaned, hits = pattern.regex.subn(
                PLACEHOLDER.format(label=pattern.label), cleaned
            )
            if hits:
                self.found[pattern.label] = (
                    self.found.get(pattern.label, 0) + hits
                )
        return cleaned

    def scrub_all(self, values: Iterable[str]) -> list[str]:
        return [self.scrub(value) for value in values]

    @property
    def clean(self) -> bool:
        return not self.found
