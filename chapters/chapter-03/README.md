# Chapter 3 — Anatomy of an LLM Application

**Part I — Foundations**

The port, the scripted client, and the first vertical slice.

## Code built in this chapter

- [`src/llmapp/llm/base.py`](../../src/llmapp/llm/base.py)
- [`src/llmapp/llm/fake.py`](../../src/llmapp/llm/fake.py)
- [`src/llmapp/pipeline.py`](../../src/llmapp/pipeline.py)

## Tests

- [`tests/test_pipeline.py`](../../tests/test_pipeline.py)

Run them with:

```bash
pytest tests/test_pipeline.py -q
```

---

Chapter code lives in the shared `llmapp` package rather than
in this directory, because the book builds one application
incrementally rather than 28 separate examples. This file is
the index into it.
