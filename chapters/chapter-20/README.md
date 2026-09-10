# Chapter 20 — Regression, Adversarial Testing, and CI/CD Quality Gates

**Part VII — Evaluation and Testing**

Baselines, model qualification, adversarial suite, CI gates.

## Code built in this chapter

- [`src/llmapp/eval/regression.py`](../../src/llmapp/eval/regression.py)
- [`src/llmapp/eval/adversarial.py`](../../src/llmapp/eval/adversarial.py)
- [`scripts/quality_gate.py`](../../scripts/quality_gate.py)
- [`evalsetup.py`](../../evalsetup.py)
- [`evals/adversarial.jsonl`](../../evals/adversarial.jsonl)

## Tests

- [`tests/test_regression_and_adversarial.py`](../../tests/test_regression_and_adversarial.py)

Run them with:

```bash
pytest tests/test_regression_and_adversarial.py -q
```

---

Chapter code lives in the shared `llmapp` package rather than
in this directory, because the book builds one application
incrementally rather than 28 separate examples. This file is
the index into it.
