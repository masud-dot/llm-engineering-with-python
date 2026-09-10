"""Lexical retrieval, promoted from a script into the package."""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from llmapp.retrieval.chunking import Chunk
from llmapp.retrieval.store import Hit

WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lower-case word and number tokens."""
    return WORD.findall(text.lower())


@dataclass
class BM25Index:
    """Exact-term retrieval over the same chunks."""

    chunks: list[Chunk] = field(default_factory=list)
    _bm25: BM25Okapi | None = None

    def add(self, chunks: Sequence[Chunk]) -> None:
        self.chunks.extend(chunks)
        self._bm25 = BM25Okapi(
            [tokenize(c.text) for c in self.chunks]
        )

    def __len__(self) -> int:
        return len(self.chunks)

    def search(
        self,
        query: str,
        k: int = 5,
        where: dict[str, str] | None = None,
    ) -> list[Hit]:
        if self._bm25 is None or not self.chunks:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        allowed = [
            i
            for i, chunk in enumerate(self.chunks)
            if _matches(chunk, where)
        ]
        allowed.sort(key=lambda i: -scores[i])
        return [
            Hit(chunk=self.chunks[i], score=float(scores[i]))
            for i in allowed[:k]
            if scores[i] > 0.0
        ]


def _matches(chunk: Chunk, where: dict[str, str] | None) -> bool:
    if not where:
        return True
    return all(
        chunk.metadata.get(key) == value
        for key, value in where.items()
    )
