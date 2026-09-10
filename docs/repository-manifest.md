# Repository manifest

What is here, why, and where to look. Maintenance document; also
useful to a reader navigating the repository for the first time.

## Top level

| Path | Purpose |
|---|---|
| `README.md` | Entry point: quick start, verified state, layout |
| `LICENSE` | MIT licence for the code in this repository |
| `pyproject.toml` | Package metadata, dependencies, console script, pytest config |
| `requirements.txt` | Lockfile generated from the tested environment |
| `.env.example` | Every environment variable the projects read |
| `.gitignore` | Excludes secrets, caches, virtual environments, data |
| `.dockerignore` | Keeps `.env`, tests, and `.git` out of image layers |
| `Dockerfile` | Two-stage build, non-root user, warmed tokenizer cache |
| `compose.yaml` | Local stack: API, PostgreSQL with pgvector, Redis |
| `evalsetup.py` | Wires the evaluation harness to the application |
| `.github/workflows/tests.yml` | Five CI jobs: offline, local-server, evaluation, build, live |

## Directories

| Path | Files | Purpose |
|---|---|---|
| `src/llmapp/` | 93 | The application. One package, built across 28 chapters |
| `tests/` | 58 | 428 tests mirroring `src/`; 388 need no credentials |
| `scripts/` | 13 | Standalone measurement and operations utilities |
| `prompts/` | 11 | Versioned prompt files; released versions are never edited |
| `evals/` | 6 | Golden datasets, the adversarial suite, stored baselines |
| `corpus/` | 4 | Sample documents, by team; the directory tree is the access model |
| `chapters/` | 28 | One index per chapter, pointing into the package |
| `projects/` | 5 | The five capstone projects, with measured results |
| `docs/` | 8 | Setup, architecture, security, troubleshooting, mappings |

## The package

| Module | Purpose | Chapters |
|---|---|---|
| `config.py` | Typed settings, validation, production rules | 4, 25 |
| `llm/` | Provider-neutral client port and adapters, reliability, cost | 3, 5, 6 |
| `prompts/` | Versioned registry, templates, context budget, memory | 7, 8 |
| `schemas/` | Strict schemas, native structured output, repair loop | 9 |
| `retrieval/` | Vectors, chunking, embedding, stores, lexical, hybrid, rerank | 10-13 |
| `rag/` | Ingestion, answering, pipeline, grounding, access control | 12-14 |
| `tools/` | Tool contracts, registry, execution loop, MCP client and server | 15, 17 |
| `agents/` | Bounded loop, state, limits, the research agent | 16 |
| `eval/` | Datasets, metrics, judge, runner, regression, adversarial | 19, 20 |
| `security/` | Redaction, output handling, the SQL gate | 21 |
| `obs/` | Logging, cache, router, quota, metrics, tracing, feedback | 6, 22, 23 |
| `api/` | Schemas, auth, limits, jobs, app factory, entry point | 24, 25 |
| `projects/` | The assembled capstones | 26-28 |

## Tests

| Marker | Meaning | Command |
|---|---|---|
| (none) | Offline. No credentials, no network after the tokenizer cache is warm | `pytest -m "not local_server and not live"` |
| `local_server` | Spawns a local HTTP or MCP process, or needs PostgreSQL | `pytest -m local_server` |
| `live` | Calls a real provider and costs money. Nightly only | `pytest -m live` |

Tests requiring PostgreSQL skip cleanly when `LLMAPP_TEST_PG_DSN` or
`LLMAPP_TEST_READER_DSN` is unset.

## Configuration

Every variable is prefixed `LLMAPP_` and documented in
`.env.example`. Model identifiers and prices are configuration, not
code, so updating them when a provider changes its catalogue is an
environment change rather than an edit.

## What is deliberately absent

- **No `.env`.** Excluded by `.gitignore` and `.dockerignore`.
- **No framework spine.** LangChain is not a dependency. `langgraph`
  is discussed in Chapter 16 and deliberately not used, so a reader
  can see the agent loop rather than a framework's version of it.
- **No `respx`.** It cannot intercept `openai` 3.8.0, which runs on
  `httpx2`; local endpoints through `base_url` replace it.
- **No generated test output.** `projects/02-test-case-generator`
  produces modules at run time rather than shipping them.
