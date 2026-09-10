"""One interface, several places to keep vectors."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from llmapp.retrieval.chunking import Chunk


@dataclass(frozen=True)
class Record:
    """A chunk and its vector, ready to store."""

    chunk: Chunk
    vector: np.ndarray


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk and its similarity score."""

    chunk: Chunk
    score: float


class VectorStore(Protocol):
    """What the retrieval pipeline needs from storage."""

    @property
    def dimensions(self) -> int: ...

    def upsert(self, records: Sequence[Record]) -> None: ...

    def search(
        self,
        vector: np.ndarray,
        k: int = 5,
        where: dict[str, str] | None = None,
    ) -> list[Hit]: ...

    def delete(self, chunk_ids: Sequence[str]) -> None: ...

    def count(self) -> int: ...


class MemoryStore:
    """Brute-force search over a NumPy matrix."""

    def __init__(self, dimensions: int) -> None:
        self._dimensions = dimensions
        self._rows: dict[str, Record] = {}

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def upsert(self, records: Sequence[Record]) -> None:
        for record in records:
            if record.vector.shape[0] != self._dimensions:
                raise ValueError("wrong number of dimensions")
            self._rows[record.chunk.chunk_id] = record

    def delete(self, chunk_ids: Sequence[str]) -> None:
        for chunk_id in chunk_ids:
            self._rows.pop(chunk_id, None)

    def count(self) -> int:
        return len(self._rows)

    def search(
        self,
        vector: np.ndarray,
        k: int = 5,
        where: dict[str, str] | None = None,
    ) -> list[Hit]:
        kept = [
            record
            for record in self._rows.values()
            if _matches(record.chunk, where)
        ]
        if not kept:
            return []
        matrix = np.vstack([r.vector for r in kept])
        scores = matrix @ vector
        order = np.argsort(-scores)[:k]
        return [
            Hit(chunk=kept[i].chunk, score=float(scores[i]))
            for i in order
        ]


def _matches(chunk: Chunk, where: dict[str, str] | None) -> bool:
    if not where:
        return True
    return all(
        chunk.metadata.get(key) == value
        for key, value in where.items()
    )
