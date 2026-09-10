# Threat model

Full treatment in Chapter 21. This is the operator's summary.

## Trust boundaries

Untrusted: end-user input, retrieved documents **whatever their
origin**, tool results, remote MCP tool descriptions, and the model's
own output.

Trusted: your prompt templates, your schemas, your tool definitions.

## Controls, strongest first

**Capability constraints** hold regardless of what the model decides:
tool allowlists, scoped credentials, per-call human approval, and
step, cost, and rate limits.

**Output validation** is deterministic: schema checks, citation
grounding, quantity verification, link allowlists, credential-shape
scanning.

**Input filtering** is a rate limiter, not a boundary. The signature
space is unbounded and legitimate content contains attack-shaped text.

## Rules that are not negotiable

- Never expose a shell, `eval`, or arbitrary-SQL tool.
- Access labels come from a system of record, never from document
  text.
- Cache keys include the principal, or one user's answer is served to
  another.
- Filter before ranking. Post-filtering removes the citation, not the
  influence.
- The database role is the boundary. Application gates reduce how
  often you rely on it.
- Guardrails live in code and fail closed.

## Verifying the boundary

```bash
pytest tests/test_security.py tests/test_injection.py -q
psql "$LLMAPP_TEST_READER_DSN" -c 'DELETE FROM shop.orders;'
```
