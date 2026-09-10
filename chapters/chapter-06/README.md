# Chapter 6 — The Reliability Layer: Retries, Timeouts, Fallbacks, Logging, and Cost

**Part II — Python and the API Layer**

Error taxonomy, backoff, circuit breaker, cost accounting.

## Code built in this chapter

- [`src/llmapp/llm/errors.py`](../../src/llmapp/llm/errors.py)
- [`src/llmapp/llm/reliable.py`](../../src/llmapp/llm/reliable.py)
- [`src/llmapp/llm/pricing.py`](../../src/llmapp/llm/pricing.py)
- [`src/llmapp/obs/log.py`](../../src/llmapp/obs/log.py)

## Tests

- [`tests/test_reliable.py`](../../tests/test_reliable.py)
- [`tests/test_reliable_integration.py`](../../tests/test_reliable_integration.py)

Run them with:

```bash
pytest tests/test_reliable.py tests/test_reliable_integration.py -q
```

---

Chapter code lives in the shared `llmapp` package rather than
in this directory, because the book builds one application
incrementally rather than 28 separate examples. This file is
the index into it.
