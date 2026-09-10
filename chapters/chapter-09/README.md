# Chapter 9 — Structured Outputs: Schemas, Pydantic, and Repair

**Part III — Prompt, Context, and Output**

Strict schemas, native structured output, bounded repair loop.

## Code built in this chapter

- [`src/llmapp/schemas/triage.py`](../../src/llmapp/schemas/triage.py)
- [`src/llmapp/schemas/strict.py`](../../src/llmapp/schemas/strict.py)
- [`src/llmapp/schemas/validate.py`](../../src/llmapp/schemas/validate.py)

## Tests

- [`tests/test_schemas.py`](../../tests/test_schemas.py)
- [`tests/test_structured_integration.py`](../../tests/test_structured_integration.py)

Run them with:

```bash
pytest tests/test_schemas.py tests/test_structured_integration.py -q
```

---

Chapter code lives in the shared `llmapp` package rather than
in this directory, because the book builds one application
incrementally rather than 28 separate examples. This file is
the index into it.
