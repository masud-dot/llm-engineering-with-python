# Chapter 13 — Advanced Retrieval: Hybrid, Reranking, Rewriting, Filtering

**Part V — Retrieval-Augmented Generation**

BM25, rank fusion, reranking, multi-query, the improvement ledger.

## Code built in this chapter

- [`src/llmapp/retrieval/lexical.py`](../../src/llmapp/retrieval/lexical.py)
- [`src/llmapp/retrieval/fusion.py`](../../src/llmapp/retrieval/fusion.py)
- [`src/llmapp/retrieval/hybrid.py`](../../src/llmapp/retrieval/hybrid.py)
- [`src/llmapp/retrieval/rerank.py`](../../src/llmapp/retrieval/rerank.py)
- [`src/llmapp/retrieval/window.py`](../../src/llmapp/retrieval/window.py)
- [`src/llmapp/rag/rewrite.py`](../../src/llmapp/rag/rewrite.py)
- [`scripts/retrieval_ledger.py`](../../scripts/retrieval_ledger.py)

## Tests

- [`tests/test_advanced_retrieval.py`](../../tests/test_advanced_retrieval.py)

Run them with:

```bash
pytest tests/test_advanced_retrieval.py -q
```

---

Chapter code lives in the shared `llmapp` package rather than
in this directory, because the book builds one application
incrementally rather than 28 separate examples. This file is
the index into it.
