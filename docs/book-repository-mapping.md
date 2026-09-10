# Book to repository mapping

Every chapter of *LLM Engineering with Python* and the code it
builds. The book explains; this repository is what it explains.

Chapter code lives in the shared `llmapp` package rather than in
per-chapter folders, because the book builds one application
incrementally. `chapters/chapter-NN/README.md` is the index into
the package for that chapter.

## Reading path

```
Book chapter
  -> chapters/chapter-NN/README.md   (what was built)
  -> src/llmapp/<module>.py          (the implementation)
  -> tests/test_<area>.py            (the behaviour, asserted)
  -> projects/<project>/README.md    (where it is assembled)
```

## Chapter map

### Part I — Foundations

| Ch | Title | Implementation | Tests |
|---|---|---|---|
| 1 | What LLM Engineering Is (and Is Not) | — | — |
| 2 | Inside the Black Box: Tokens, Context, Sampling, and Limits | `scripts/count_tokens.py` | — |
| 3 | Anatomy of an LLM Application | `src/llmapp/llm/base.py`<br>`src/llmapp/llm/fake.py`<br>`src/llmapp/pipeline.py` | `tests/test_pipeline.py` |
### Part II — Python and the API Layer

| Ch | Title | Implementation | Tests |
|---|---|---|---|
| 4 | Environment, Configuration, and Secrets | `src/llmapp/config.py`<br>`src/llmapp/llm/budget.py`<br>`scripts/verify_setup.py`<br>`scripts/scan_secrets.py`<br>`.env.example` | `tests/test_config.py`<br>`tests/test_budget.py` |
| 5 | Talking to Models: Requests, Responses, Streaming, and Async | `src/llmapp/llm/openai_adapter.py`<br>`src/llmapp/llm/anthropic_adapter.py`<br>`src/llmapp/llm/factory.py`<br>`src/llmapp/llm/conversation.py` | `tests/test_openai_adapter.py`<br>`tests/test_anthropic_adapter.py`<br>`tests/test_capabilities.py`<br>`tests/test_conversation.py` |
| 6 | The Reliability Layer: Retries, Timeouts, Fallbacks, Logging, and Cost | `src/llmapp/llm/errors.py`<br>`src/llmapp/llm/reliable.py`<br>`src/llmapp/llm/pricing.py`<br>`src/llmapp/obs/log.py` | `tests/test_reliable.py`<br>`tests/test_reliable_integration.py` |
### Part III — Prompt, Context, and Output Engineering

| Ch | Title | Implementation | Tests |
|---|---|---|---|
| 7 | Prompt Engineering for Engineers | `src/llmapp/prompts/registry.py`<br>`src/llmapp/prompts/chain.py`<br>`prompts/triage.v1.toml`<br>`prompts/triage.v2.toml`<br>`scripts/compare_prompts.py` | `tests/test_prompts.py`<br>`tests/test_chain.py` |
| 8 | Context Engineering: Budgets, Compaction, and Memory | `src/llmapp/prompts/context.py`<br>`src/llmapp/prompts/memory.py`<br>`scripts/memory_cost.py` | `tests/test_context.py`<br>`tests/test_memory.py` |
| 9 | Structured Outputs: Schemas, Pydantic, and Repair | `src/llmapp/schemas/triage.py`<br>`src/llmapp/schemas/strict.py`<br>`src/llmapp/schemas/validate.py` | `tests/test_schemas.py`<br>`tests/test_structured_integration.py` |
### Part IV — Embeddings and Retrieval

| Ch | Title | Implementation | Tests |
|---|---|---|---|
| 10 | Embeddings and Semantic Search | `src/llmapp/retrieval/vectors.py`<br>`src/llmapp/retrieval/chunking.py`<br>`src/llmapp/retrieval/embed.py`<br>`src/llmapp/retrieval/search.py`<br>`src/llmapp/retrieval/metrics.py`<br>`scripts/compare_retrieval.py` | `tests/test_vectors.py`<br>`tests/test_chunking.py`<br>`tests/test_embed_search.py` |
| 11 | Vector Stores and Retrieval Pipelines | `src/llmapp/retrieval/store.py`<br>`src/llmapp/retrieval/chroma_store.py`<br>`src/llmapp/retrieval/pg_store.py`<br>`scripts/index_tradeoff.py` | `tests/test_stores.py` |
### Part V — Retrieval-Augmented Generation

| Ch | Title | Implementation | Tests |
|---|---|---|---|
| 12 | Building a RAG System End to End | `src/llmapp/rag/ingest.py`<br>`src/llmapp/rag/answer.py`<br>`src/llmapp/rag/pipeline.py`<br>`prompts/rag_answer.v1.toml` | `tests/test_ingest.py`<br>`tests/test_rag_pipeline.py` |
| 13 | Advanced Retrieval: Hybrid, Reranking, Rewriting, Filtering | `src/llmapp/retrieval/lexical.py`<br>`src/llmapp/retrieval/fusion.py`<br>`src/llmapp/retrieval/hybrid.py`<br>`src/llmapp/retrieval/rerank.py`<br>`src/llmapp/retrieval/window.py`<br>`src/llmapp/rag/rewrite.py`<br>`scripts/retrieval_ledger.py` | `tests/test_advanced_retrieval.py` |
| 14 | Trustworthy RAG: Grounding, Citations, and Failure Modes | `src/llmapp/rag/grounding.py`<br>`src/llmapp/rag/access.py` | `tests/test_trustworthy_rag.py`<br>`tests/test_failure_modes.py` |
### Part VI — Tools and Agents

| Ch | Title | Implementation | Tests |
|---|---|---|---|
| 15 | Tool Calling: Contracts, Execution, and Safety | `src/llmapp/tools/base.py`<br>`src/llmapp/tools/registry.py`<br>`src/llmapp/tools/loop.py` | `tests/test_tools.py`<br>`tests/test_tools_integration.py` |
| 16 | Agents: Loops, Planning, Memory, and Control | `src/llmapp/agents/state.py`<br>`src/llmapp/agents/loop.py`<br>`src/llmapp/agents/research.py`<br>`prompts/agent_plan.v1.toml`<br>`prompts/agent_step.v1.toml` | `tests/test_agent.py`<br>`tests/test_research_agent.py` |
| 17 | Standardizing Tool Access with MCP | `src/llmapp/tools/mcp_server.py`<br>`src/llmapp/tools/mcp_client.py`<br>`scripts/mcp_tools_server.py` | `tests/test_mcp.py` |
### Part VII — Evaluation and Testing

| Ch | Title | Implementation | Tests |
|---|---|---|---|
| 18 | Testing Nondeterministic Systems | `tests/_cassette.py`<br>`tests/_statistics.py`<br>`.github/workflows/tests.yml` | `tests/test_testing_strategy.py` |
| 19 | Evaluating LLM and RAG Output | `src/llmapp/eval/dataset.py`<br>`src/llmapp/eval/metrics.py`<br>`src/llmapp/eval/judge.py`<br>`src/llmapp/eval/runner.py`<br>`prompts/eval_judge.v1.toml`<br>`prompts/eval_pairwise.v1.toml` | `tests/test_eval.py` |
| 20 | Regression, Adversarial Testing, and CI/CD Quality Gates | `src/llmapp/eval/regression.py`<br>`src/llmapp/eval/adversarial.py`<br>`scripts/quality_gate.py`<br>`evalsetup.py`<br>`evals/adversarial.jsonl` | `tests/test_regression_and_adversarial.py` |
### Part VIII — Security, Cost, and Observability

| Ch | Title | Implementation | Tests |
|---|---|---|---|
| 21 | Security: Prompt Injection, Data Leakage, and Tool Safety | `src/llmapp/security/redact.py`<br>`src/llmapp/security/output.py`<br>`src/llmapp/security/sql.py` | `tests/test_security.py`<br>`tests/test_injection.py` |
| 22 | Cost, Latency, Caching, and Model Routing | `src/llmapp/obs/cache.py`<br>`src/llmapp/obs/router.py`<br>`src/llmapp/obs/quota.py`<br>`scripts/cost_report.py` | `tests/test_cost.py` |
| 23 | Observability: Logging, Tracing, Metrics, and Alerting | `src/llmapp/obs/tracing.py`<br>`src/llmapp/obs/metrics.py`<br>`src/llmapp/obs/feedback.py` | `tests/test_observability.py` |
### Part IX — Deployment

| Ch | Title | Implementation | Tests |
|---|---|---|---|
| 24 | Serving LLM Applications with FastAPI | `src/llmapp/api/schemas.py`<br>`src/llmapp/api/security.py`<br>`src/llmapp/api/limits.py`<br>`src/llmapp/api/jobs.py`<br>`src/llmapp/api/main.py` | `tests/test_api.py` |
| 25 | Packaging, Docker, Configuration, and CI/CD | `pyproject.toml`<br>`requirements.txt`<br>`Dockerfile`<br>`.dockerignore`<br>`compose.yaml`<br>`src/llmapp/api/run.py`<br>`src/llmapp/retrieval/local.py`<br>`scripts/readiness_audit.py` | `tests/test_deployment.py` |
### Part X — Capstone Projects

| Ch | Title | Implementation | Tests |
|---|---|---|---|
| 26 | Project 1: The AI Document Assistant | `src/llmapp/projects/assistant.py`<br>`corpus/`<br>`evals/assistant.jsonl` | `tests/test_document_assistant.py` |
| 27 | Project 2: The AI Test-Case Generator | `src/llmapp/projects/testgen/` | `tests/test_testgen.py` |
| 28 | Project 3: The AI SQL Assistant, and Project 5: The Production Service | `src/llmapp/projects/sqlassist/`<br>`prompts/sql_assist.v1.toml`<br>`src/llmapp/api/main.py` | `tests/test_sql_assistant.py` |

## Project map

| Project | Chapter | Directory | Implementation |
|---|---|---|---|
| 1 — AI Document Assistant | 26 | `projects/01-document-assistant` | `src/llmapp/projects/assistant.py` |
| 2 — AI Test-Case Generator | 27 | `projects/02-test-case-generator` | `src/llmapp/projects/testgen/` |
| 3 — AI SQL Assistant | 28 | `projects/03-sql-assistant` | `src/llmapp/projects/sqlassist/` |
| 4 — AI Research Agent | 16 | `projects/04-research-agent` | `src/llmapp/agents/research.py` |
| 5 — Production LLM Service | 28 | `projects/05-production-service` | `src/llmapp/api/` |

> **Numbering note.** Project numbers follow the book, in which
> Project 2 is the Test-Case Generator and Project 3 is the SQL
> Assistant. Directory names match the book rather than any other
> ordering, so a reader moving from a chapter to the repository
> finds the number they just read.

## Documentation map

| Topic | Document | Chapters |
|---|---|---|
| Environment and installation | `docs/setup/environment.md` | 4, 25 |
| Database setup and the read-only role | `docs/setup/database.md` | 11, 28 |
| Layers, ports, dependency rule | `docs/architecture/overview.md` | 3, 5 |
| Threat model and controls | `docs/security/threat-model.md` | 21 |
| Symptoms and causes | `docs/troubleshooting.md` | all |
| Figure worklist | `docs/diagram-inventory.md` | all |

## Commands referenced in the book

Every command below runs against this repository as laid out.

```bash
python scripts/verify_setup.py            # Ch 4
python scripts/scan_secrets.py            # Ch 4
python scripts/count_tokens.py            # Ch 2
python scripts/compare_prompts.py         # Ch 7
python scripts/memory_cost.py             # Ch 8
python scripts/compare_retrieval.py       # Ch 10
python scripts/index_tradeoff.py          # Ch 11
python scripts/retrieval_ledger.py        # Ch 13
python scripts/mcp_tools_server.py        # Ch 17
python scripts/quality_gate.py --baseline main   # Ch 20
python scripts/quality_gate.py --adversarial     # Ch 20
python scripts/cost_report.py             # Ch 22
python scripts/readiness_audit.py         # Ch 25
python -m llmapp.api.run                  # Ch 24, 25
```
