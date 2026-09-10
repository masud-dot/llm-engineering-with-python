# Chapter 8 — Context Engineering: Budgets, Compaction, and Memory

**Part III — Prompt, Context, and Output**

Token budget allocator and four memory strategies.

## Code built in this chapter

- [`src/llmapp/prompts/context.py`](../../src/llmapp/prompts/context.py)
- [`src/llmapp/prompts/memory.py`](../../src/llmapp/prompts/memory.py)
- [`scripts/memory_cost.py`](../../scripts/memory_cost.py)

## Tests

- [`tests/test_context.py`](../../tests/test_context.py)
- [`tests/test_memory.py`](../../tests/test_memory.py)

Run them with:

```bash
pytest tests/test_context.py tests/test_memory.py -q
```

---

Chapter code lives in the shared `llmapp` package rather than
in this directory, because the book builds one application
incrementally rather than 28 separate examples. This file is
the index into it.
