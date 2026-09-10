"""Split documents into retrievable pieces."""

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """A retrievable piece, and where it came from."""

    text: str
    source: str
    ordinal: int
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return f"{self.source}#{self.ordinal}"


def split_sentences(text: str) -> list[str]:
    """A deliberately simple sentence splitter."""
    parts = SENTENCE_END.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    source: str,
    *,
    max_words: int = 120,
    overlap_words: int = 20,
    metadata: dict[str, str] | None = None,
) -> list[Chunk]:
    """Group whole sentences into overlapping chunks."""
    if overlap_words >= max_words:
        raise ValueError("overlap must be smaller than max_words")

    chunks: list[Chunk] = []
    current: list[str] = []
    size = 0
    for sentence in split_sentences(text):
        words = len(sentence.split())
        if current and size + words > max_words:
            chunks.append(
                _build(current, source, len(chunks), metadata)
            )
            current, size = _carry(current, overlap_words)
        current.append(sentence)
        size += words
    if current:
        chunks.append(
            _build(current, source, len(chunks), metadata)
        )
    return chunks


def _build(
    sentences: list[str],
    source: str,
    ordinal: int,
    metadata: dict[str, str] | None,
) -> Chunk:
    return Chunk(
        text=" ".join(sentences),
        source=source,
        ordinal=ordinal,
        metadata=dict(metadata or {}),
    )


def _carry(
    sentences: list[str], overlap_words: int
) -> tuple[list[str], int]:
    """Keep trailing sentences as the next chunk's lead-in."""
    kept: list[str] = []
    size = 0
    for sentence in reversed(sentences):
        words = len(sentence.split())
        if size + words > overlap_words:
            break
        kept.insert(0, sentence)
        size += words
    return kept, size


def iter_chunks(
    documents: dict[str, str], **kwargs: object
) -> Iterator[Chunk]:
    for source, text in documents.items():
        yield from chunk_text(
            text, source, **kwargs  # type: ignore[arg-type]
        )
