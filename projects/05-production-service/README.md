# Project 5 — Production LLM Service

**Built in Chapter 28.** Implementation: [`src/llmapp/api/`](../../src/llmapp/api/) · Tests: [`tests/test_api.py`](../../tests/test_api.py)

All three assistants behind one authenticated, traced, rate-limited
HTTP service.

## What it demonstrates

- Endpoints for synchronous, streaming, and background execution
- JWT authentication producing the principal that scopes retrieval
- Per-principal rate limiting and per-tenant quotas
- Liveness and readiness probes, with readiness checking the index
- Errors that carry no internals; detail goes to the trace
- Features enabled by configuration; absence returns 503

## Run it

```bash
pytest tests/test_api.py tests/test_deployment.py -q
python -m llmapp.api.run          # or: llmapp-serve
python scripts/readiness_audit.py
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | Liveness; no dependency checks |
| GET | `/readyz` | Readiness; checks the index and dependencies |
| POST | `/v1/ask` | Grounded answer with citations |
| POST | `/v1/ask/stream` | Server-sent events |
| POST | `/v1/ask/async` | Background job |
| GET | `/v1/jobs/{job_id}` | Job status; not-yours returns 404 |
| POST | `/v1/sql` | Read-only database question |
| POST | `/v1/feedback` | Graded feedback capture |
