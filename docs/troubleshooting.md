# Troubleshooting

Symptom, likely cause, chapter.

| Symptom | Cause | See |
|---|---|---|
| `Missing credentials` at startup | A factory constructed a hosted client with `LLMAPP_PROVIDER=fake`; the embedder factory needs the same provider branch as the model factory | Ch. 25 |
| Every question is refused | A relevance floor calibrated on cosine similarity applied to fused rank scores; move the floor onto each retrieval arm | Ch. 26 |
| `/readyz` returns 503 | The index loaded zero chunks; check `LLMAPP_CORPUS_DIR` and that the path resolves from the process working directory | Ch. 24 |
| Tests fail with `no cassette` | A request was made that was never recorded; record it or fix the request | Ch. 18 |
| `respx` intercepts nothing | `openai` 3.8.0 runs on `httpx2`; use a local endpoint through `base_url` instead | Ch. 6, 18 |
| An MCP server will not start | The child process gets only the environment you pass it | Ch. 17 |
| `ModuleNotFoundError: mcp.server.fastmcp` | `mcp` 2.x renamed `FastMCP` to `MCPServer` | Ch. 17 |
| BM25 returns nothing | On a corpus of a few documents, IDF collapses to zero | Ch. 13 |
| A citation fails on a correct answer | Chunk text keeps the source's line breaks; `locate()` collapses whitespace before matching | Ch. 14, 26 |
| Cache hit rate is far lower than expected | Keys include the principal, which is correct and costs hit rate | Ch. 22 |
| Retries take far longer than the policy suggests | The SDK retries twice by default on top of yours; set `max_retries=0` | Ch. 6 |
| A prompt file is ignored | Two files declare the same `name`; the registry now refuses this | Ch. 19 |
