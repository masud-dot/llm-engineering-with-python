"""Chroma backend. Embedded, no server, good for CI."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from llmapp.retrieval.chunking import Chunk
from llmapp.retrieval.store import Hit, Record

RESERVED = ("source", "ordinal")


class ChromaStore:
    """Store vectors in a local Chroma collection."""

    def __init__(
        self,
        dimensions: int,
        *,
        path: Path | None = None,
        collection: str = "llmapp_chunks",
    ) -> None:
        import chromadb

        self._dimensions = dimensions
        client: Any = (
            chromadb.PersistentClient(path=str(path))
            if path is not None
            else chromadb.EphemeralClient()
        )
        # Chroma stores distances; cosine distance is 1 - cosine
        # similarity, which search() converts back.
        self._collection = client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def upsert(self, records: Sequence[Record]) -> None:
        if not records:
            return
        self._collection.upsert(
            ids=[r.chunk.chunk_id for r in records],
            embeddings=[r.vector.tolist() for r in records],
            documents=[r.chunk.text for r in records],
            metadatas=[_flatten(r.chunk) for r in records],
        )

    def delete(self, chunk_ids: Sequence[str]) -> None:
        if chunk_ids:
            self._collection.delete(ids=list(chunk_ids))

    def count(self) -> int:
        return int(self._collection.count())

    def search(
        self,
        vector: np.ndarray,
        k: int = 5,
        where: dict[str, str] | None = None,
    ) -> list[Hit]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[vector.tolist()],
            n_results=k,
            where=_where_clause(where),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[Hit] = []
        ids = result["ids"][0]
        for position, chunk_id in enumerate(ids):
            meta = dict(result["metadatas"][0][position])
            hits.append(
                Hit(
                    chunk=Chunk(
                        text=result["documents"][0][position],
                        source=str(meta.pop("source", "")),
                        ordinal=int(meta.pop("ordinal", 0)),
                        metadata={
                            k2: str(v) for k2, v in meta.items()
                        },
                    ),
                    score=1.0 - float(
                        result["distances"][0][position]
                    ),
                )
            )
        return hits


def _flatten(chunk: Chunk) -> dict[str, str | int]:
    for key in RESERVED:
        if key in chunk.metadata:
            raise ValueError(f"{key!r} is reserved metadata")
    flat: dict[str, str | int] = dict(chunk.metadata)
    flat["source"] = chunk.source
    flat["ordinal"] = chunk.ordinal
    return flat


def _where_clause(
    where: dict[str, str] | None,
) -> dict[str, Any] | None:
    if not where:
        return None
    if len(where) == 1:
        key, value = next(iter(where.items()))
        return {key: value}
    return {"$and": [{k: v} for k, v in where.items()]}
