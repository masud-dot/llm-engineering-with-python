# Chapter 26 — Project 1: The AI Document Assistant

**Part X — Capstone Projects**

Project 1 assembled: ingestion, hybrid retrieval, permissions, evaluation.

## Code built in this chapter

- [`src/llmapp/projects/assistant.py`](../../src/llmapp/projects/assistant.py)
- [`corpus/`](../../corpus/)
- [`evals/assistant.jsonl`](../../evals/assistant.jsonl)

## Tests

- [`tests/test_document_assistant.py`](../../tests/test_document_assistant.py)

Run them with:

```bash
pytest tests/test_document_assistant.py -q
```

---

Chapter code lives in the shared `llmapp` package rather than
in this directory, because the book builds one application
incrementally rather than 28 separate examples. This file is
the index into it.
