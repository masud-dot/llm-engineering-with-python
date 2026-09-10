"""One contract, run against every backend."""

import os
from collections.abc import Iterator

import numpy as np
import pytest

from llmapp.retrieval.chunking import Chunk
from llmapp.retrieval.store import MemoryStore, Record, VectorStore

DIMS = 4
PG_DSN = os.environ.get("LLMAPP_TEST_PG_DSN", "")
try:  # the driver is an optional extra
    import psycopg  # noqa: F401
    _HAS_PSYCOPG = True
except ImportError:
    _HAS_PSYCOPG = False


def unit(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    return vector / np.linalg.norm(vector)


REFUND = Record(
    chunk=Chunk("refund policy", "a.md", 0, {"team": "billing"}),
    vector=unit(1, 0, 0, 0),
)
DELIVERY = Record(
    chunk=Chunk("delivery policy", "b.md", 0, {"team": "ship"}),
    vector=unit(0, 1, 0, 0),
)


def make_chroma() -> VectorStore:
    from llmapp.retrieval.chroma_store import ChromaStore

    return ChromaStore(DIMS)


def make_pg() -> VectorStore:
    from llmapp.retrieval.pg_store import PgVectorStore

    store = PgVectorStore(
        PG_DSN, dimensions=DIMS, model="test-embed"
    )
    store.delete([REFUND.chunk.chunk_id, DELIVERY.chunk.chunk_id])
    return store


BACKENDS = [
    pytest.param(lambda: MemoryStore(DIMS), id="memory"),
    pytest.param(make_chroma, id="chroma"),
    pytest.param(
        make_pg,
        id="pgvector",
        marks=pytest.mark.skipif(
            not (PG_DSN and _HAS_PSYCOPG),
            reason="needs LLMAPP_TEST_PG_DSN and the postgres extra"
        ),
    ),
]


@pytest.fixture(params=BACKENDS)
def store(request: pytest.FixtureRequest) -> Iterator[VectorStore]:
    built: VectorStore = request.param()
    yield built
    built.delete([REFUND.chunk.chunk_id, DELIVERY.chunk.chunk_id])


def test_a_new_store_is_empty(store: VectorStore) -> None:
    assert store.count() == 0
    assert store.search(unit(1, 0, 0, 0)) == []


def test_upsert_then_count(store: VectorStore) -> None:
    store.upsert([REFUND, DELIVERY])
    assert store.count() == 2


def test_search_ranks_by_similarity(store: VectorStore) -> None:
    store.upsert([REFUND, DELIVERY])
    hits = store.search(unit(1, 0, 0, 0), k=2)
    assert [h.chunk.chunk_id for h in hits] == ["a.md#0", "b.md#0"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[1].score == pytest.approx(0.0, abs=1e-5)


def test_provenance_survives_a_round_trip(
    store: VectorStore,
) -> None:
    store.upsert([REFUND])
    hit = store.search(unit(1, 0, 0, 0), k=1)[0]
    assert hit.chunk.source == "a.md"
    assert hit.chunk.ordinal == 0
    assert hit.chunk.metadata["team"] == "billing"


def test_metadata_filters_the_candidates(
    store: VectorStore,
) -> None:
    store.upsert([REFUND, DELIVERY])
    hits = store.search(
        unit(1, 0, 0, 0), k=5, where={"team": "ship"}
    )
    assert [h.chunk.chunk_id for h in hits] == ["b.md#0"]


def test_upsert_replaces_rather_than_duplicates(
    store: VectorStore,
) -> None:
    store.upsert([REFUND])
    changed = Record(
        chunk=Chunk(
            "refund policy, revised", "a.md", 0,
            {"team": "billing"},
        ),
        vector=unit(1, 0, 0, 0),
    )
    store.upsert([changed])
    assert store.count() == 1
    hit = store.search(unit(1, 0, 0, 0), k=1)[0]
    assert hit.chunk.text == "refund policy, revised"


def test_delete_removes_the_row(store: VectorStore) -> None:
    store.upsert([REFUND, DELIVERY])
    store.delete(["a.md#0"])
    assert store.count() == 1
    assert store.search(unit(1, 0, 0, 0), k=5)[0].chunk.source == (
        "b.md"
    )


def test_deleting_something_absent_is_harmless(
    store: VectorStore,
) -> None:
    store.delete(["nothing.md#9"])
    assert store.count() == 0
