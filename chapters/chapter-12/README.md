# Chapter 12 — Building a RAG System End to End

**Part V — Retrieval-Augmented Generation**

Ingestion, retrieval, assembly, cited generation, abstention.

## Code built in this chapter

- [`src/llmapp/rag/ingest.py`](../../src/llmapp/rag/ingest.py)
- [`src/llmapp/rag/answer.py`](../../src/llmapp/rag/answer.py)
- [`src/llmapp/rag/pipeline.py`](../../src/llmapp/rag/pipeline.py)
- [`prompts/rag_answer.v1.toml`](../../prompts/rag_answer.v1.toml)

## Tests

- [`tests/test_ingest.py`](../../tests/test_ingest.py)
- [`tests/test_rag_pipeline.py`](../../tests/test_rag_pipeline.py)

Run them with:

```bash
pytest tests/test_ingest.py tests/test_rag_pipeline.py -q
```

---

Chapter code lives in the shared `llmapp` package rather than
in this directory, because the book builds one application
incrementally rather than 28 separate examples. This file is
the index into it.
