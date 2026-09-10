# Chapter 10 — Embeddings and Semantic Search

**Part IV — Embeddings and Retrieval**

Cosine similarity from primitives, chunking, recall@k and MRR.

## Code built in this chapter

- [`src/llmapp/retrieval/vectors.py`](../../src/llmapp/retrieval/vectors.py)
- [`src/llmapp/retrieval/chunking.py`](../../src/llmapp/retrieval/chunking.py)
- [`src/llmapp/retrieval/embed.py`](../../src/llmapp/retrieval/embed.py)
- [`src/llmapp/retrieval/search.py`](../../src/llmapp/retrieval/search.py)
- [`src/llmapp/retrieval/metrics.py`](../../src/llmapp/retrieval/metrics.py)
- [`scripts/compare_retrieval.py`](../../scripts/compare_retrieval.py)

## Tests

- [`tests/test_vectors.py`](../../tests/test_vectors.py)
- [`tests/test_chunking.py`](../../tests/test_chunking.py)
- [`tests/test_embed_search.py`](../../tests/test_embed_search.py)

Run them with:

```bash
pytest tests/test_vectors.py tests/test_chunking.py tests/test_embed_search.py -q
```

---

Chapter code lives in the shared `llmapp` package rather than
in this directory, because the book builds one application
incrementally rather than 28 separate examples. This file is
the index into it.
