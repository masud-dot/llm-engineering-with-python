# Chapter 15 — Tool Calling: Contracts, Execution, and Safety

**Part VI — Tools and Agents**

Tool schemas, validation gate, approval, parallel execution.

## Code built in this chapter

- [`src/llmapp/tools/base.py`](../../src/llmapp/tools/base.py)
- [`src/llmapp/tools/registry.py`](../../src/llmapp/tools/registry.py)
- [`src/llmapp/tools/loop.py`](../../src/llmapp/tools/loop.py)

## Tests

- [`tests/test_tools.py`](../../tests/test_tools.py)
- [`tests/test_tools_integration.py`](../../tests/test_tools_integration.py)

Run them with:

```bash
pytest tests/test_tools.py tests/test_tools_integration.py -q
```

---

Chapter code lives in the shared `llmapp` package rather than
in this directory, because the book builds one application
incrementally rather than 28 separate examples. This file is
the index into it.
