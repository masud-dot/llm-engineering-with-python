# Database setup

PostgreSQL is optional. Without it, 40 tests are deselected and the
rest pass.

## Vector store (Chapter 11)

```bash
createdb llmapp
psql -d llmapp -c 'CREATE EXTENSION IF NOT EXISTS vector;'
export LLMAPP_TEST_PG_DSN=postgresql://user:pass@127.0.0.1:5432/llmapp
```

## SQL assistant (Chapter 28)

The assistant executes as a role that cannot write. That role is the
project's real security boundary, so create it first.

```sql
CREATE SCHEMA shop;
-- tables: shop.customers, shop.orders, shop.salaries

CREATE ROLE llmapp_reader LOGIN PASSWORD '<choose one>';
GRANT USAGE ON SCHEMA shop TO llmapp_reader;
GRANT SELECT ON shop.customers, shop.orders TO llmapp_reader;
```

`shop.salaries` is deliberately left ungranted: the tests assert the
role refuses to read it.

```bash
export LLMAPP_TEST_READER_DSN=postgresql://llmapp_reader:...@127.0.0.1:5432/llmapp
pytest tests/test_sql_assistant.py -q
```

## Verifying the boundary

Before trusting the assistant, confirm the role refuses writes:

```bash
psql "$LLMAPP_TEST_READER_DSN" -c 'DELETE FROM shop.orders;'
# ERROR:  permission denied for table orders
```
