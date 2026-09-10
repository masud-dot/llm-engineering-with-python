# Chapter 23 — Observability: Logging, Tracing, Metrics, and Alerting

**Part VIII — Security, Cost, Observability**

OpenTelemetry spans, quality proxies, alert rules, feedback capture.

## Code built in this chapter

- [`src/llmapp/obs/tracing.py`](../../src/llmapp/obs/tracing.py)
- [`src/llmapp/obs/metrics.py`](../../src/llmapp/obs/metrics.py)
- [`src/llmapp/obs/feedback.py`](../../src/llmapp/obs/feedback.py)

## Tests

- [`tests/test_observability.py`](../../tests/test_observability.py)

Run them with:

```bash
pytest tests/test_observability.py -q
```

---

Chapter code lives in the shared `llmapp` package rather than
in this directory, because the book builds one application
incrementally rather than 28 separate examples. This file is
the index into it.
