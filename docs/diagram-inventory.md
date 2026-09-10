# Diagram inventory

**71 figures across 28 chapters.** Every figure is specified
in the manuscript at its point of use, with a purpose, a component
list, and a placement instruction. This inventory is the production
worklist; the manuscript is the specification.

## Production notes

- All figures are line diagrams intended for **greyscale print** at
  6 x 9 inches. No figure may depend on colour to carry meaning;
  use line weight, dash patterns, and fill texture instead.
- Figure 1.1 is the book's master diagram. Figures 3.1 and 28.2 are
  detailed expansions of it and must stay visually consistent with
  it — same box shapes, same layer ordering, same labels.
- Chapter numbers appear inside component boxes in Figures 26.1 and
  28.2 so a reader can navigate from the map to the chapter.
- Trust boundaries are drawn as dashed lines throughout (Figures
  1.1, 17.2, 21.1, 21.2, 28.1).
- Any figure carrying measured numbers (13.x, 22.3, 26.1) must match
  the manuscript transcripts exactly; Phase 5 re-verifies both.

## Inventory

| Figure | Part | Chapter | Title | Purpose |
|---|---|---|---|---|
| 1.1 | I | 1 | Reference architecture of an LLM application | The book's master diagram, referenced by every Part. |
| 1.2 | I | 1 | The LLM engineering loop | Establish the book's organizing lifecycle and show that it is a loop rather than a pipeline. |
| 2.1 | I | 2 | From text to tokens to model input | Show the pipeline that every request passes through, and locate where token counting happens. |
| 2.2 | I | 2 | Context budget allocation | Show the window as a fixed budget divided among competing claims, so that "we have a large window" is understood as a spending decision rather than fr |
| 2.3 | I | 2 | Prefill and decode on a latency timeline | Explain why two requests with the same total token count can feel completely different to a user. |
| 3.1 | I | 3 | Layered architecture of the reference application | The chapter's anchor diagram and the detailed version of Figure 1.1. |
| 3.2 | I | 3 | Request lifecycle sequence | Show the twelve steps as a sequence across layers, with failure branches. |
| 3.3 | I | 3 | Project structure and its map to the layers | Connect directories to architecture so that the reader knows where new code belongs. |
| 4.1 | II | 4 | Configuration resolution order | Show which source wins when the same setting is defined more than once. |
| 5.1 | II | 5 | Request and response flow with timing | Show what crosses the wire and where time is spent, joining Chapter 2's prefill/decode model to the code in this chapter. |
| 5.2 | II | 5 | Streaming event sequence | Show what arrives over a streamed response and which events an application consumes. |
| 5.3 | II | 5 | Provider-neutral client and adapters | Show the one place vendor code lives and what depends on what. |
| 6.1 | II | 6 | Error taxonomy decision tree | Convert an incoming failure into one of three responses without guesswork. |
| 6.2 | II | 6 | Retry, backoff, and breaker state machine | Show the full control flow of one logical call. |
| 6.3 | II | 6 | The degradation ladder | Present graceful degradation as a designed sequence rather than an accident. |
| 7.1 | III | 7 | Prompt anatomy and trust | Show the four parts, who writes each, and where the trust boundary falls. |
| 7.2 | III | 7 | Prompt chain with intermediate artifacts | Show decomposition as a pipeline of small, checkable steps. |
| 7.3 | III | 7 | Prompt registry and version flow | Show the lifecycle of a prompt change. |
| 8.1 | III | 8 | Context budget allocation | Show the window as a fixed budget divided among competing claims, and show what happens when the claims exceed it. |
| 8.2 | III | 8 | Four memory strategies compared | Show growth, cost, and fact retention side by side. |
| 9.1 | III | 9 | Structured output pipeline with the repair loop | Show the full path from request to typed object, including every exit. |
| 9.2 | III | 9 | From Pydantic model to validated object | Show the single source of truth flowing in both directions. |
| 10.1 | IV | 10 | From text to vector to similarity | Make the pipeline concrete before any code. |
| 10.2 | IV | 10 | Chunking strategies compared | Show the same document under four strategies and the retrieval consequence of each. |
| 10.3 | IV | 10 | The same queries under both methods | Show complementary failure modes rather than a winner. |
| 11.1 | IV | 11 | Vector store architecture | Show what a store contains and where each part is used. |
| 11.2 | IV | 11 | Recall against latency for three index types | Present the trade-off as a curve with a visible knee, and show the effect of data structure. |
| 11.3 | IV | 11 | The retrieval pipeline with a swappable backend | Show where the seam falls and what crosses it. |
| 12.1 | V | 12 | Full RAG architecture | The chapter's map, and the diagram Chapters 13 and 14 extend. |
| 12.2 | V | 12 | The ingestion pipeline | Show a single file's journey and where each hazard lives. |
| 12.3 | V | 12 | Context assembly with provenance | Show retrieved hits becoming a budgeted, labeled passage block. |
| 13.1 | V | 13 | Hybrid retrieval with rank fusion | Show two incomparable rankings becoming one. |
| 13.2 | V | 13 | Retrieve, rerank, generate | Show the funnel and where the cost sits. |
| 13.3 | V | 13 | Multi-query fan-out | Show one question becoming several searches and one fused result. |
| 14.1 | V | 14 | RAG failure-mode map | Attach each failure to the stage that produces it and the control that catches it. |
| 14.2 | V | 14 | Permission-aware retrieval | Show where authorization is applied and why it cannot be skipped. |
| 15.1 | VI | 15 | The tool-calling sequence | Show the round trip and mark where the application's authority sits. |
| 15.2 | VI | 15 | The validation and execution gate | Show every check a call passes before a handler runs. |
| 15.3 | VI | 15 | Parallel tool execution | Show bounded concurrency with order preserved. |
| 16.1 | VI | 16 | The agent loop | Show the cycle and every exit from it. |
| 16.2 | VI | 16 | Agent state across steps | Show what persists, what is re-sent, and what grows. |
| 16.3 | VI | 16 | Human-in-the-loop approval | Show suspension and resumption as one path with an outside decision. |
| 16.4 | VI | 16 | Research agent architecture | Show Project 4 end to end with its controls marked. |
| 17.1 | VI | 17 | N × M versus a standard interface | Show the integration count collapsing. |
| 17.2 | VI | 17 | MCP client and server with the trust boundary | Show what crosses the boundary in each direction. |
| 18.1 | VII | 18 | The LLM test pyramid | Show the layers with their real counts and costs. |
| 18.2 | VII | 18 | Record and replay | Show one mechanism in two modes. |
| 18.3 | VII | 18 | Test tiers and when each runs | Show what runs where, with the real counts and the resources each needs. |
| 19.1 | VII | 19 | The evaluation pipeline | Show the whole flow from dataset to decision. |
| 19.2 | VII | 19 | Judge calibration | Show the loop that makes a judge trustworthy. |
| 19.3 | VII | 19 | RAG evaluated stage by stage | Show which metric belongs to which stage and what each rules out. |
| 20.1 | VII | 20 | The pipeline with quality gates | Show what runs when, and what each can block. |
| 20.2 | VII | 20 | Model qualification | Show the procedure as a gated sequence. |
| 20.3 | VII | 20 | The production feedback loop | Show the cycle that turns incidents into permanent protection. |
| 21.1 | VIII | 21 | Threat model with trust boundaries | Show where trust changes and what crosses each boundary. |
| 21.2 | VIII | 21 | An indirect injection attack path | Trace the payload from authoring to containment. |
| 21.3 | VIII | 21 | Layered defenses | Show the three guardrail categories with their strength and placement. |
| 22.1 | VIII | 22 | Where cost and latency go | Show both budgets decomposed for one request. |
| 22.2 | VIII | 22 | Cache layers |  |
| 22.3 | VIII | 22 | Model routing with escalation | Show the ladder and where the cost accrues. |
| 23.1 | VIII | 23 | A trace waterfall for one request | Show a multi-step request decomposed in time, with the attributes that matter attached. |
| 23.2 | VIII | 23 | The observability stack | Show the three signal types, what each answers, and where they come from. |
| 23.3 | VIII | 23 | The alerting decision tree | Route a firing signal to the right response. |
| 24.1 | IX | 24 | Service architecture | Show the request path through the edge and into the application. |
| 24.2 | IX | 24 | Streaming and background job flows | Contrast the three execution shapes on one timeline. |
| 25.1 | IX | 25 | The build and deploy pipeline | Show the path from commit to production with every gate marked. |
| 25.2 | IX | 25 | Container topology | Show what is in the image, what is mounted, and what is injected. |
| 26.1 | X | 26 | Document Assistant architecture | The project's map, showing which chapter owns each component. |
| 27.1 | X | 27 | Test-case generator architecture | Show which stages are deterministic and which involve a model. |
| 28.1 | X | 28 | SQL assistant architecture, with every gate | Show the five checks between a question and a row, and mark which are application controls and which are database controls. |
| 28.2 | X | 28 | The assembled production service | The book's final diagram — every component, with the chapter that built it. |

## Figures per chapter

| Chapter | Figures |
|---|---|
| 1 | 2 |
| 2 | 3 |
| 3 | 3 |
| 4 | 1 |
| 5 | 3 |
| 6 | 3 |
| 7 | 3 |
| 8 | 2 |
| 9 | 2 |
| 10 | 3 |
| 11 | 3 |
| 12 | 3 |
| 13 | 3 |
| 14 | 2 |
| 15 | 3 |
| 16 | 4 |
| 17 | 2 |
| 18 | 3 |
| 19 | 3 |
| 20 | 3 |
| 21 | 3 |
| 22 | 3 |
| 23 | 3 |
| 24 | 2 |
| 25 | 2 |
| 26 | 1 |
| 27 | 1 |
| 28 | 2 |

**Total: 71**
