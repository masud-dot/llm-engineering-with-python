# Chapter 19 — Evaluating LLM and RAG Output

**Part VII — Evaluation and Testing**

Golden datasets, nDCG, judge calibration, the evaluation gate.

## Code built in this chapter

- [`src/llmapp/eval/dataset.py`](../../src/llmapp/eval/dataset.py)
- [`src/llmapp/eval/metrics.py`](../../src/llmapp/eval/metrics.py)
- [`src/llmapp/eval/judge.py`](../../src/llmapp/eval/judge.py)
- [`src/llmapp/eval/runner.py`](../../src/llmapp/eval/runner.py)
- [`prompts/eval_judge.v1.toml`](../../prompts/eval_judge.v1.toml)
- [`prompts/eval_pairwise.v1.toml`](../../prompts/eval_pairwise.v1.toml)

## Tests

- [`tests/test_eval.py`](../../tests/test_eval.py)

Run them with:

```bash
pytest tests/test_eval.py -q
```

---

Chapter code lives in the shared `llmapp` package rather than
in this directory, because the book builds one application
incrementally rather than 28 separate examples. This file is
the index into it.
