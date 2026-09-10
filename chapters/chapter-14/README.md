# Chapter 14 — Trustworthy RAG: Grounding, Citations, and Failure Modes

**Part V — Retrieval-Augmented Generation**

Span citations, quantity checks, conflicts, enforced permissions.

## Code built in this chapter

- [`src/llmapp/rag/grounding.py`](../../src/llmapp/rag/grounding.py)
- [`src/llmapp/rag/access.py`](../../src/llmapp/rag/access.py)

## Tests

- [`tests/test_trustworthy_rag.py`](../../tests/test_trustworthy_rag.py)
- [`tests/test_failure_modes.py`](../../tests/test_failure_modes.py)

Run them with:

```bash
pytest tests/test_trustworthy_rag.py tests/test_failure_modes.py -q
```

---

Chapter code lives in the shared `llmapp` package rather than
in this directory, because the book builds one application
incrementally rather than 28 separate examples. This file is
the index into it.
