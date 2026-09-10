# Project 2 — AI Test-Case Generator

**Built in Chapter 27.** Implementation: [`src/llmapp/projects/testgen/`](../../src/llmapp/projects/testgen/) · Tests: [`tests/test_testgen.py`](../../tests/test_testgen.py)

Turns an OpenAPI specification into reviewable, runnable pytest
modules. Demonstrated against this repository's own API.

## What it demonstrates

- A specification treated as untrusted, bounded input
- A coverage plan written in code, not chosen by the model
- Structured generation with a bounded repair loop
- Code emission gated by `ast.parse`, with values JSON-encoded
- A self-review pass tested by feeding it deliberate defects

## Run it

```bash
pytest tests/test_testgen.py -q
```

## Measured result

7 endpoints, 33 cases planned deterministically, 7 modules emitted
and collected by pytest. Executed against the running service:

| configuration | result |
|---|---|
| as generated | 29/33 |
| declared statuses + constraints respected | 31/33 |
| with a corrected fixture | 32/33 |

The self-review pass reported perfect coverage while four
assertions were wrong. Structural checks are not semantic checks.
