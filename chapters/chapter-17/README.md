# Chapter 17 — Standardizing Tool Access with MCP

**Part VI — Tools and Agents**

Exposing and consuming tools over the Model Context Protocol.

## Code built in this chapter

- [`src/llmapp/tools/mcp_server.py`](../../src/llmapp/tools/mcp_server.py)
- [`src/llmapp/tools/mcp_client.py`](../../src/llmapp/tools/mcp_client.py)
- [`scripts/mcp_tools_server.py`](../../scripts/mcp_tools_server.py)

## Tests

- [`tests/test_mcp.py`](../../tests/test_mcp.py)

Run them with:

```bash
pytest tests/test_mcp.py -q
```

---

Chapter code lives in the shared `llmapp` package rather than
in this directory, because the book builds one application
incrementally rather than 28 separate examples. This file is
the index into it.
