# Chapter 28 — Project 3: The AI SQL Assistant, and Project 5: The Production Service

**Part X — Capstone Projects**

Projects 3 and 5: read-only SQL with a role boundary, then full assembly.

## Code built in this chapter

- [`src/llmapp/projects/sqlassist/`](../../src/llmapp/projects/sqlassist/)
- [`prompts/sql_assist.v1.toml`](../../prompts/sql_assist.v1.toml)
- [`src/llmapp/api/main.py`](../../src/llmapp/api/main.py)

## Tests

- [`tests/test_sql_assistant.py`](../../tests/test_sql_assistant.py)

Run them with:

```bash
pytest tests/test_sql_assistant.py -q
```

---

Chapter code lives in the shared `llmapp` package rather than
in this directory, because the book builds one application
incrementally rather than 28 separate examples. This file is
the index into it.
