# Chapter 7 — Prompt Engineering for Engineers

**Part III — Prompt, Context, and Output**

Versioned prompt files, injection-safe rendering, A/B comparison.

## Code built in this chapter

- [`src/llmapp/prompts/registry.py`](../../src/llmapp/prompts/registry.py)
- [`src/llmapp/prompts/chain.py`](../../src/llmapp/prompts/chain.py)
- [`prompts/triage.v1.toml`](../../prompts/triage.v1.toml)
- [`prompts/triage.v2.toml`](../../prompts/triage.v2.toml)
- [`scripts/compare_prompts.py`](../../scripts/compare_prompts.py)

## Tests

- [`tests/test_prompts.py`](../../tests/test_prompts.py)
- [`tests/test_chain.py`](../../tests/test_chain.py)

Run them with:

```bash
pytest tests/test_prompts.py tests/test_chain.py -q
```

---

Chapter code lives in the shared `llmapp` package rather than
in this directory, because the book builds one application
incrementally rather than 28 separate examples. This file is
the index into it.
