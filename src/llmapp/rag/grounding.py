"""Check that an answer is supported by the passages."""

import re
from dataclasses import dataclass

from llmapp.rag.answer import Citation, GroundedAnswer
from llmapp.retrieval.store import Hit

SENTENCE = re.compile(r"(?<=[.!?])\s+")
WORD = re.compile(r"[a-z0-9]+")
NUMERIC = re.compile(r"\d[\d,.]*")

# Word-overlap alone cannot see a single substituted number,
# so quantities are checked separately and strictly.
NUMBER_WORDS = frozenset(
    """zero one two three four five six seven eight nine ten
    eleven twelve thirteen fourteen fifteen sixteen seventeen
    eighteen nineteen twenty thirty forty fifty sixty seventy
    eighty ninety hundred thousand""".split()
)


@dataclass(frozen=True)
class Span:
    """Where a quote sits inside the chunk it came from."""

    chunk_id: str
    start: int
    end: int
    quote: str


@dataclass(frozen=True)
class GroundingReport:
    """How well the answer is tied to its evidence."""

    spans: list[Span]
    unsupported: list[str]
    coverage: float
    conflicts: list[str]

    @property
    def is_grounded(self) -> bool:
        return bool(self.spans) and not self.unsupported


def _collapse(text: str) -> str:
    return " ".join(text.split())


def locate(quote: str, text: str, chunk_id: str) -> Span | None:
    """Find a verbatim quote and record its offsets.

    Chunk text keeps the line breaks the source had, so a
    model quoting a passage it reads as one line fails a raw
    substring check. Whitespace is collapsed on both sides
    before matching; everything else must match exactly.
    """
    position = text.find(quote)
    if position == -1:
        flat_text, flat_quote = _collapse(text), _collapse(quote)
        flat_position = flat_text.find(flat_quote)
        if flat_position == -1:
            return None
        return Span(
            chunk_id=chunk_id,
            start=flat_position,
            end=flat_position + len(flat_quote),
            quote=flat_quote,
        )
    return Span(
        chunk_id=chunk_id,
        start=position,
        end=position + len(quote),
        quote=quote,
    )


def quantities(text: str) -> set[str]:
    """Numbers and number words, which must be supported."""
    lowered = text.lower()
    found = set(NUMERIC.findall(lowered))
    found |= {
        word
        for word in WORD.findall(lowered)
        if word in NUMBER_WORDS
    }
    return found


def _content_words(text: str) -> set[str]:
    return {
        word
        for word in WORD.findall(text.lower())
        if len(word) > 3
    }


def check(
    answer: GroundedAnswer,
    hits: list[Hit],
    *,
    min_overlap: float = 0.5,
) -> GroundingReport:
    """Verify citations, then check every sentence is covered."""
    by_id = {h.chunk.chunk_id: h.chunk.text for h in hits}
    spans: list[Span] = []
    for citation in answer.citations:
        body = by_id.get(citation.chunk_id)
        if body is None:
            continue
        found = locate(citation.quote, body, citation.chunk_id)
        if found is not None:
            spans.append(found)

    evidence = " ".join(
        by_id[span.chunk_id] for span in spans
    ).lower()
    evidence_words = _content_words(evidence)

    unsupported: list[str] = []
    sentences = [
        s.strip()
        for s in SENTENCE.split(answer.answer)
        if s.strip()
    ]
    for sentence in sentences:
        words = _content_words(sentence)
        if not words:
            continue
        shared = len(words & evidence_words) / len(words)
        invented = quantities(sentence) - quantities(evidence)
        if shared < min_overlap or invented:
            unsupported.append(sentence)

    covered = len(sentences) - len(unsupported)
    coverage = covered / len(sentences) if sentences else 0.0
    return GroundingReport(
        spans=spans,
        unsupported=unsupported,
        coverage=coverage,
        conflicts=detect_conflicts(hits),
    )


def detect_conflicts(hits: list[Hit]) -> list[str]:
    """Flag passages on one topic with different dates."""
    by_topic: dict[str, set[str]] = {}
    for hit in hits:
        topic = hit.chunk.metadata.get("topic")
        date = hit.chunk.metadata.get("effective_date")
        if topic and date:
            by_topic.setdefault(topic, set()).add(date)
    return sorted(
        topic
        for topic, dates in by_topic.items()
        if len(dates) > 1
    )


def demote(
    answer: GroundedAnswer, report: GroundingReport, reason: str
) -> GroundedAnswer:
    """Turn an unsupported answer into an honest refusal."""
    return GroundedAnswer(
        answered=False,
        answer="",
        citations=list(answer.citations)[:0],
        missing=reason,
    )


def keep_verified(
    answer: GroundedAnswer, report: GroundingReport
) -> GroundedAnswer:
    """Drop citations that did not resolve."""
    verified = {span.chunk_id: span.quote for span in report.spans}
    kept = [
        Citation(chunk_id=cid, quote=quote)
        for cid, quote in verified.items()
    ]
    return answer.model_copy(update={"citations": kept})
