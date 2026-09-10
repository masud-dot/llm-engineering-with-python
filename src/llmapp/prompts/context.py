"""The context window is a budget. Spend it deliberately."""

from dataclasses import dataclass, field

import tiktoken

DEFAULT_ENCODING = "o200k_base"


def count_tokens(text: str, encoding: str = DEFAULT_ENCODING) -> int:
    """Exact token count for a piece of context."""
    return len(tiktoken.get_encoding(encoding).encode(text))


def truncate_to(
    text: str, limit: int, encoding: str = DEFAULT_ENCODING
) -> str:
    """Cut text to at most `limit` tokens, on a token edge."""
    if limit <= 0:
        return ""
    enc = tiktoken.get_encoding(encoding)
    pieces = enc.encode(text)
    if len(pieces) <= limit:
        return text
    return enc.decode(pieces[:limit])


@dataclass(frozen=True)
class Section:
    """One claim on the window."""

    name: str
    content: str
    priority: int = 0  # higher survives longer
    required: bool = False
    truncatable: bool = False


@dataclass(frozen=True)
class FitResult:
    """What survived the budget, and what did not."""

    sections: list[Section]
    dropped: list[str] = field(default_factory=list)
    truncated: list[str] = field(default_factory=list)
    tokens_used: int = 0

    def text(self) -> str:
        return "\n\n".join(s.content for s in self.sections)


class BudgetExceeded(ValueError):
    """Required sections alone do not fit the window."""


@dataclass(frozen=True)
class ContextBudget:
    """Split a window between competing claims."""

    window_tokens: int
    reserved_output: int = 512
    encoding: str = DEFAULT_ENCODING

    @property
    def available(self) -> int:
        return self.window_tokens - self.reserved_output

    def fit(self, sections: list[Section]) -> FitResult:
        """Keep as much as fits, dropping lowest priority first."""
        ordered = sorted(
            sections,
            key=lambda s: (s.required, s.priority),
            reverse=True,
        )
        budget = self.available
        required_cost = sum(
            count_tokens(s.content, self.encoding)
            for s in ordered
            if s.required
        )
        if required_cost > budget:
            raise BudgetExceeded(
                f"required sections need {required_cost} tokens "
                f"of {budget} available"
            )

        kept: list[Section] = []
        dropped: list[str] = []
        truncated: list[str] = []
        spent = 0
        for section in ordered:
            cost = count_tokens(section.content, self.encoding)
            if spent + cost <= budget:
                kept.append(section)
                spent += cost
                continue
            room = budget - spent
            if section.truncatable and room > 0:
                shorter = truncate_to(
                    section.content, room, self.encoding
                )
                kept.append(
                    Section(
                        name=section.name,
                        content=shorter,
                        priority=section.priority,
                        required=section.required,
                        truncatable=True,
                    )
                )
                truncated.append(section.name)
                spent += count_tokens(shorter, self.encoding)
                continue
            dropped.append(section.name)

        order = {s.name: i for i, s in enumerate(sections)}
        kept.sort(key=lambda s: order[s.name])
        return FitResult(
            sections=kept,
            dropped=dropped,
            truncated=truncated,
            tokens_used=spent,
        )
