# Chapter 21 — Security: Prompt Injection, Data Leakage, and Tool Safety

**Part VIII — Security, Cost, Observability**

Threat model, redaction, URL policy, SQL gate, poisoned corpus.

## Code built in this chapter

- [`src/llmapp/security/redact.py`](../../src/llmapp/security/redact.py)
- [`src/llmapp/security/output.py`](../../src/llmapp/security/output.py)
- [`src/llmapp/security/sql.py`](../../src/llmapp/security/sql.py)

## Tests

- [`tests/test_security.py`](../../tests/test_security.py)
- [`tests/test_injection.py`](../../tests/test_injection.py)

Run them with:

```bash
pytest tests/test_security.py tests/test_injection.py -q
```

---

Chapter code lives in the shared `llmapp` package rather than
in this directory, because the book builds one application
incrementally rather than 28 separate examples. This file is
the index into it.
