# Project 4 — AI Research Agent

**Built in Chapter 16.** Implementation: [`src/llmapp/agents/research.py`](../../src/llmapp/agents/research.py) · Tests: [`tests/test_research_agent.py`](../../tests/test_research_agent.py)

A bounded agent that researches a question against the corpus,
writes a brief, and publishes it only with human approval.

## What it demonstrates

- A written plan before any tool runs
- Three independent limits: steps, tool calls, and spend
- Stall detection by call signature, with a nudge before stopping
- Approval that binds to the exact call a human was shown
- An auditable run record

## Run it

```bash
pytest tests/test_research_agent.py -q
```

## Note

Approval resumption executes the stored call rather than asking the
model again. The regression test scripts a substituted publish call
after the approved one and asserts the substitute never runs.
