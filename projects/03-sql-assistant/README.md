# Project 3 — AI SQL Assistant

**Built in Chapter 28.** Implementation: [`src/llmapp/projects/sqlassist/`](../../src/llmapp/projects/sqlassist/) · Tests: [`tests/test_sql_assistant.py`](../../tests/test_sql_assistant.py)

Natural language to a validated read-only SQL query, executed under
a database role that cannot write.

## What it demonstrates

- Schema introspection through the read-only connection
- A JSON contract with an explicit abstention path
- The Chapter 21 SQL gate: one statement, SELECT only, allowlisted
  tables, injected row limit
- An `EXPLAIN` dry run that validates columns and prices the query
- A database role as the boundary no application bug can bypass

## Run it

Requires PostgreSQL. See `docs/setup/database.md`.

```bash
export LLMAPP_TEST_READER_DSN=postgresql://llmapp_reader:...@host/db
pytest tests/test_sql_assistant.py -q
```

## Measured result

Seven questions, five refusals, each naming a different control —
including an invented column that passed the regex gate and was
caught by the query planner. Bypassing every application control,
the read-only role refused `DELETE`, `UPDATE`, an ungranted
`SELECT`, and `DROP`.
