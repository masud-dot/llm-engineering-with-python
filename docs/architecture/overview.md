# Architecture

The repository is one application built incrementally across 28
chapters, not 28 separate examples. Layers depend downward only, and
provider SDKs are imported in exactly one directory.

## Layers

| Layer | Package | Chapters |
|---|---|---|
| Interface | `llmapp.api` | 24, 25 |
| Orchestration | `llmapp.rag`, `llmapp.agents`, `llmapp.projects` | 12–16, 26–28 |
| Model access | `llmapp.llm` | 3, 5, 6 |
| Prompts and context | `llmapp.prompts` | 7, 8 |
| Output contracts | `llmapp.schemas` | 9 |
| Retrieval | `llmapp.retrieval` | 10–13 |
| Tools | `llmapp.tools` | 15, 17 |
| Evaluation | `llmapp.eval` | 19, 20 |
| Security | `llmapp.security` | 21 |
| Observability and cost | `llmapp.obs` | 6, 22, 23 |
| Configuration | `llmapp.config` | 4 |

## The dependency rule

Dependencies point downward. A layer may use the layer below and must
not know about the layer above. The practical consequence:
`openai` and `anthropic` are imported only in `llmapp/llm/`, so a
provider migration touches one directory and the 300-plus tests above
it do not move.

## Ports

Four protocols carry the load:

- `LLMClient` — send messages, receive text with usage and a stop
  reason. Optional capabilities (`SupportsStreaming`, `SupportsAsync`,
  `SupportsTools`, `SupportsStructuredOutput`) are separate protocols
  so an adapter advertises what it implements.
- `Embedder` — text to vectors, with a declared width.
- `VectorStore` — five methods, three backends, one contract suite.
- `SearchIndex` — what `RagPipeline` needs from a retriever, so a
  semantic index and a hybrid retriever are interchangeable.
