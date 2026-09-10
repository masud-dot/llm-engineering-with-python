"""Semantic search in one small class."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from llmapp.retrieval.chunking import Chunk
from llmapp.retrieval.embed import Embedder
from llmapp.retrieval.store import Hit, MemoryStore, Record, VectorStore
from llmapp.retrieval.vectors import normalize


__all__ = ["Hit", "SemanticIndex"]


class SemanticIndex:
    """Embed on the way in, search on the way out.

    The storage is a VectorStore, so the same pipeline runs
    against memory, Chroma, or Postgres unchanged.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore | None = None,
    ) -> None:
        if store is not None and (
            store.dimensions != embedder.dimensions
        ):
            raise ValueError(
                "store and embedder disagree on dimensions"
            )
        self._embedder = embedder
        self._store = store or MemoryStore(embedder.dimensions)

    def __len__(self) -> int:
        return self._store.count()

    @property
    def size(self) -> int:
        """Public chunk count, for health checks."""
        return self._store.count()

    def add(self, chunks: Sequence[Chunk]) -> None:
        """Embed and store. Vectors are normalized once."""
        if not chunks:
            return
        vectors = normalize(
            self._embedder.embed([c.text for c in chunks])
        )
        self._store.upsert(
            [
                Record(chunk=chunk, vector=vectors[i])
                for i, chunk in enumerate(chunks)
            ]
        )

    def remove(self, chunk_ids: Sequence[str]) -> None:
        self._store.delete(chunk_ids)

    def search(
        self,
        query: str,
        k: int = 5,
        where: dict[str, str] | None = None,
    ) -> list[Hit]:
        """Rank chunks against the query, filtered first."""
        vector = normalize(self._embedder.embed([query]))[0]
        return self._store.search(vector, k=k, where=where)
