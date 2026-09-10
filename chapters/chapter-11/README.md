# Chapter 11 — Vector Stores and Retrieval Pipelines

**Part IV — Embeddings and Retrieval**

One store protocol, three backends, one contract suite.

## Code built in this chapter

- [`src/llmapp/retrieval/store.py`](../../src/llmapp/retrieval/store.py)
- [`src/llmapp/retrieval/chroma_store.py`](../../src/llmapp/retrieval/chroma_store.py)
- [`src/llmapp/retrieval/pg_store.py`](../../src/llmapp/retrieval/pg_store.py)
- [`scripts/index_tradeoff.py`](../../scripts/index_tradeoff.py)

## Tests

- [`tests/test_stores.py`](../../tests/test_stores.py)

Run them with:

```bash
pytest tests/test_stores.py -q
```

---

Chapter code lives in the shared `llmapp` package rather than
in this directory, because the book builds one application
incrementally rather than 28 separate examples. This file is
the index into it.
