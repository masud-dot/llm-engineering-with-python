# Chapter 5 — Talking to Models: Requests, Responses, Streaming, and Async

**Part II — Python and the API Layer**

Provider-neutral port with adapters, streaming, async, capabilities.

## Code built in this chapter

- [`src/llmapp/llm/openai_adapter.py`](../../src/llmapp/llm/openai_adapter.py)
- [`src/llmapp/llm/anthropic_adapter.py`](../../src/llmapp/llm/anthropic_adapter.py)
- [`src/llmapp/llm/factory.py`](../../src/llmapp/llm/factory.py)
- [`src/llmapp/llm/conversation.py`](../../src/llmapp/llm/conversation.py)

## Tests

- [`tests/test_openai_adapter.py`](../../tests/test_openai_adapter.py)
- [`tests/test_anthropic_adapter.py`](../../tests/test_anthropic_adapter.py)
- [`tests/test_capabilities.py`](../../tests/test_capabilities.py)
- [`tests/test_conversation.py`](../../tests/test_conversation.py)

Run them with:

```bash
pytest tests/test_openai_adapter.py tests/test_anthropic_adapter.py tests/test_capabilities.py tests/test_conversation.py -q
```

---

Chapter code lives in the shared `llmapp` package rather than
in this directory, because the book builds one application
incrementally rather than 28 separate examples. This file is
the index into it.
