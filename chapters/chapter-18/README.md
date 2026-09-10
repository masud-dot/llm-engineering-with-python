# Chapter 18 — Testing Nondeterministic Systems

**Part VII — Evaluation and Testing**

Test doubles, cassettes, property tests, CI tiers.

## Code built in this chapter

- [`tests/_cassette.py`](../../tests/_cassette.py)
- [`tests/_statistics.py`](../../tests/_statistics.py)
- [`.github/workflows/tests.yml`](../../.github/workflows/tests.yml)

## Tests

- [`tests/test_testing_strategy.py`](../../tests/test_testing_strategy.py)

Run them with:

```bash
pytest tests/test_testing_strategy.py -q
```

---

Chapter code lives in the shared `llmapp` package rather than
in this directory, because the book builds one application
incrementally rather than 28 separate examples. This file is
the index into it.
