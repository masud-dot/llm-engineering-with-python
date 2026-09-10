# LLM Engineering with Python — companion repository

The complete, running reference application built across the 28
chapters of *LLM Engineering with Python: Build, Test, Secure, and
Deploy Production-Ready LLM Applications with Python, RAG, and AI
Agents*, by Masud Mondal.

This is one application built incrementally, not 28 separate
examples. Every chapter adds a module to the same package.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e ".[dev]"
# optional: PostgreSQL backends and the FAISS measurement script
# pip install -e ".[postgres,bench]"
cp .env.example .env
python scripts/verify_setup.py
pytest -q -m "not local_server and not live"
```

**No API key is required** for that last command, or for any test in
this repository. Nothing here calls a paid API.

## Verified state

| Check | Result |
|---|---|
| Core install, no extras, no databases | 406 passed, 22 skipped |
| Offline tier only | 379 passed, 9 skipped, 40 deselected |
| With `[postgres]` and a local PostgreSQL | 428 passed |
| Type check | `mypy --strict`, no issues in 107 source files |
| Python | 3.12 (verified on 3.13) |

Tests that need PostgreSQL skip cleanly when the driver or the
connection string is absent — they never fail or error.

## Layout

```
chapters/     one index per chapter, pointing into the package
projects/     the five capstone projects, with measured results
src/llmapp/   the application itself
tests/        428 tests, mirroring src/
prompts/      versioned prompt files (name.vN.toml)
evals/        golden datasets and the adversarial suite
corpus/       the sample document corpus, by team
scripts/      standalone utilities and the quality gate
docs/         setup, architecture, security, troubleshooting
```

## The five projects

| # | Project | Chapter | Directory |
|---|---|---|---|
| 1 | AI Document Assistant | 26 | [`projects/01-document-assistant`](projects/01-document-assistant) |
| 2 | AI Test-Case Generator | 27 | [`projects/02-test-case-generator`](projects/02-test-case-generator) |
| 3 | AI SQL Assistant | 28 | [`projects/03-sql-assistant`](projects/03-sql-assistant) |
| 4 | AI Research Agent | 16 | [`projects/04-research-agent`](projects/04-research-agent) |
| 5 | Production LLM Service | 28 | [`projects/05-production-service`](projects/05-production-service) |

## Version baseline

Every example was executed against the pinned versions in
`requirements.txt`, on Python 3.12, in September 2026. The core:

```
openai==3.8.0        anthropic==1.4.0      pydantic==2.13.5
fastapi==0.141.1     chromadb==1.5.9       mcp==2.1.1
tiktoken==0.14.0     pytest==9.1.1         opentelemetry-sdk==1.44.0
```

Model identifiers and prices appear only in `.env.example` and in the
book's Appendix B, never in code, so updating them is a configuration
change.

This ecosystem moves. If an SDK has changed since the book went to
print, this repository is the living copy — check the issues and the
commit history before assuming the book is wrong.

## Running the tiers

```bash
pytest -q -m "not local_server and not live"   # offline, no key
pytest -q -m "local_server"                     # spawns local servers
pytest -q                                       # everything available
mypy --strict src/llmapp scripts evalsetup.py
python scripts/quality_gate.py --baseline main  # needs a real provider
python scripts/quality_gate.py --adversarial    # needs a real provider
python scripts/readiness_audit.py               # production checklist
```

## Documentation

- [Environment setup](docs/setup/environment.md)
- [Database setup](docs/setup/database.md)
- [Architecture](docs/architecture/overview.md)
- [Threat model](docs/security/threat-model.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Repository manifest](docs/repository-manifest.md)
- [Book-to-repository mapping](docs/book-repository-mapping.md)

## Licence

MIT. See [`LICENSE`](LICENSE). The code is free to use, modify, and
redistribute, including commercially, with attribution and no
warranty. This licence covers the code in this repository only; the
book's text is copyrighted separately, as stated on its copyright
page.

## Errata and questions

Open an issue. Corrections to the book are tracked there, and the
test suite is the fastest way to confirm whether a behaviour has
changed under a dependency update.
