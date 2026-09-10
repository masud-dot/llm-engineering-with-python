# Project 1 — AI Document Assistant

**Built in Chapter 26.** Implementation: [`src/llmapp/projects/assistant.py`](../../src/llmapp/projects/assistant.py) · Tests: [`tests/test_document_assistant.py`](../../tests/test_document_assistant.py)

Answers policy questions from a permissioned document corpus with
verified citations, refusing when the corpus does not cover the
question.

## What it demonstrates

- Format-aware ingestion (Markdown, HTML, PDF) with provenance
- Access labels derived from directory placement, never from text
- Hybrid retrieval with a floor on each arm before rank fusion
- Grounded answers with span-level, verified citations
- Conflict detection across superseded policy versions
- A seven-case evaluation dataset with an improvement ledger

## Run it

```bash
pytest tests/test_document_assistant.py -q
```

## Measured result

| configuration | pass | recall | ndcg | precision | abstention |
|---|---|---|---|---|---|
| semantic only | 86% | 0.80 | 0.73 | 0.57 | 86% |
| hybrid, no floors | 43% | 1.00 | 0.93 | 0.35 | 71% |
| hybrid + floors | 86% | 1.00 | 0.93 | 0.67 | 86% |

The middle row is the finding: hybrid retrieval raised recall and
halved the pass rate. One open failure (`platinum-tier`) remains,
deliberately unresolved and set as an exercise.
