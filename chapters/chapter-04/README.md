# Chapter 4 — Environment, Configuration, and Secrets

**Part II — Python and the API Layer**

Typed settings, spend ceiling, environment verification.

## Code built in this chapter

- [`src/llmapp/config.py`](../../src/llmapp/config.py)
- [`src/llmapp/llm/budget.py`](../../src/llmapp/llm/budget.py)
- [`scripts/verify_setup.py`](../../scripts/verify_setup.py)
- [`scripts/scan_secrets.py`](../../scripts/scan_secrets.py)
- [`.env.example`](../../.env.example)

## Tests

- [`tests/test_config.py`](../../tests/test_config.py)
- [`tests/test_budget.py`](../../tests/test_budget.py)

Run them with:

```bash
pytest tests/test_config.py tests/test_budget.py -q
```

---

Chapter code lives in the shared `llmapp` package rather than
in this directory, because the book builds one application
incrementally rather than 28 separate examples. This file is
the index into it.
