# Chapter 24 — Serving LLM Applications with FastAPI

**Part IX — Deployment**

Authenticated endpoints, streaming, jobs, health and readiness.

## Code built in this chapter

- [`src/llmapp/api/schemas.py`](../../src/llmapp/api/schemas.py)
- [`src/llmapp/api/security.py`](../../src/llmapp/api/security.py)
- [`src/llmapp/api/limits.py`](../../src/llmapp/api/limits.py)
- [`src/llmapp/api/jobs.py`](../../src/llmapp/api/jobs.py)
- [`src/llmapp/api/main.py`](../../src/llmapp/api/main.py)

## Tests

- [`tests/test_api.py`](../../tests/test_api.py)

Run them with:

```bash
pytest tests/test_api.py -q
```

---

Chapter code lives in the shared `llmapp` package rather than
in this directory, because the book builds one application
incrementally rather than 28 separate examples. This file is
the index into it.
