# Chapter 25 — Packaging, Docker, Configuration, and CI/CD

**Part IX — Deployment**

Reproducible build, container, pipeline, readiness audit.

## Code built in this chapter

- [`pyproject.toml`](../../pyproject.toml)
- [`requirements.txt`](../../requirements.txt)
- [`Dockerfile`](../../Dockerfile)
- [`.dockerignore`](../../.dockerignore)
- [`compose.yaml`](../../compose.yaml)
- [`src/llmapp/api/run.py`](../../src/llmapp/api/run.py)
- [`src/llmapp/retrieval/local.py`](../../src/llmapp/retrieval/local.py)
- [`scripts/readiness_audit.py`](../../scripts/readiness_audit.py)

## Tests

- [`tests/test_deployment.py`](../../tests/test_deployment.py)

Run them with:

```bash
pytest tests/test_deployment.py -q
```

---

Chapter code lives in the shared `llmapp` package rather than
in this directory, because the book builds one application
incrementally rather than 28 separate examples. This file is
the index into it.
