# Chapter 22 — Cost, Latency, Caching, and Model Routing

**Part VIII — Security, Cost, Observability**

Principal-scoped caching, escalation routing, quotas, attribution.

## Code built in this chapter

- [`src/llmapp/obs/cache.py`](../../src/llmapp/obs/cache.py)
- [`src/llmapp/obs/router.py`](../../src/llmapp/obs/router.py)
- [`src/llmapp/obs/quota.py`](../../src/llmapp/obs/quota.py)
- [`scripts/cost_report.py`](../../scripts/cost_report.py)

## Tests

- [`tests/test_cost.py`](../../tests/test_cost.py)

Run them with:

```bash
pytest tests/test_cost.py -q
```

---

Chapter code lives in the shared `llmapp` package rather than
in this directory, because the book builds one application
incrementally rather than 28 separate examples. This file is
the index into it.
