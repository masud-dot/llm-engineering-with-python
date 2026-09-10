# Chapter 16 — Agents: Loops, Planning, Memory, and Control

**Part VI — Tools and Agents**

Bounded agent loop, stall detection, resumable approval. Project 4.

## Code built in this chapter

- [`src/llmapp/agents/state.py`](../../src/llmapp/agents/state.py)
- [`src/llmapp/agents/loop.py`](../../src/llmapp/agents/loop.py)
- [`src/llmapp/agents/research.py`](../../src/llmapp/agents/research.py)
- [`prompts/agent_plan.v1.toml`](../../prompts/agent_plan.v1.toml)
- [`prompts/agent_step.v1.toml`](../../prompts/agent_step.v1.toml)

## Tests

- [`tests/test_agent.py`](../../tests/test_agent.py)
- [`tests/test_research_agent.py`](../../tests/test_research_agent.py)

Run them with:

```bash
pytest tests/test_agent.py tests/test_research_agent.py -q
```

---

Chapter code lives in the shared `llmapp` package rather than
in this directory, because the book builds one application
incrementally rather than 28 separate examples. This file is
the index into it.
