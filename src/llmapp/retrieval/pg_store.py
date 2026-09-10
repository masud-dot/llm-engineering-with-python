"""Postgres backend. For teams that already run Postgres."""

import json
from collections.abc import Sequence

import numpy as np

from llmapp.retrieval.chunking import Chunk
from llmapp.retrieval.store import Hit, Record

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   text PRIMARY KEY,
    source     text        NOT NULL,
    ordinal    integer     NOT NULL,
    body       text        NOT NULL,
    metadata   jsonb       NOT NULL DEFAULT '{}',
    model      text        NOT NULL,
    embedding  vector(%(dims)s) NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_metadata_idx
    ON chunks USING gin (metadata);
"""

INDEX = """
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
"""


class PgVectorStore:
    """Vectors, metadata, and documents in one database."""

    def __init__(
        self, dsn: str, *, dimensions: int, model: str
    ) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self._dimensions = dimensions
        self._model = model
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._conn.execute(SCHEMA % {"dims": dimensions})
        register_vector(self._conn)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def create_index(self) -> None:
        """Build the ANN index after bulk loading, not before."""
        self._conn.execute(INDEX)

    def upsert(self, records: Sequence[Record]) -> None:
        if not records:
            return
        rows = [
            (
                r.chunk.chunk_id,
                r.chunk.source,
                r.chunk.ordinal,
                r.chunk.text,
                json.dumps(r.chunk.metadata),
                self._model,
                np.asarray(r.vector, dtype=np.float32),
            )
            for r in records
        ]
        with self._conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (chunk_id, source, ordinal,
                    body, metadata, model, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    body = EXCLUDED.body,
                    metadata = EXCLUDED.metadata,
                    model = EXCLUDED.model,
                    embedding = EXCLUDED.embedding,
                    updated_at = now()
                """,
                rows,
            )

    def delete(self, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        self._conn.execute(
            "DELETE FROM chunks WHERE chunk_id = ANY(%s)",
            (list(chunk_ids),),
        )

    def count(self) -> int:
        row = self._conn.execute(
            "SELECT count(*) FROM chunks"
        ).fetchone()
        return int(row[0]) if row else 0

    def search(
        self,
        vector: np.ndarray,
        k: int = 5,
        where: dict[str, str] | None = None,
    ) -> list[Hit]:
        # The <=> operator is cosine distance, so similarity
        # is 1 - distance. Filtering happens in the WHERE
        # clause, before the ANN scan orders results.
        sql = """
            SELECT chunk_id, source, ordinal, body, metadata,
                   1 - (embedding <=> %s) AS similarity
            FROM chunks
            WHERE metadata @> %s
            ORDER BY embedding <=> %s
            LIMIT %s
        """
        params = (
            np.asarray(vector, dtype=np.float32),
            json.dumps(where or {}),
            np.asarray(vector, dtype=np.float32),
            k,
        )
        rows = self._conn.execute(sql, params).fetchall()
        return [
            Hit(
                chunk=Chunk(
                    text=row[3],
                    source=row[1],
                    ordinal=row[2],
                    metadata=dict(row[4]),
                ),
                score=float(row[5]),
            )
            for row in rows
        ]
