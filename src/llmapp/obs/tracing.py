"""Tracing an LLM request, with privacy built into the span.

Attribute names follow the OpenTelemetry generative-AI
convention (the gen_ai.* namespace) so that traces are
readable by tooling that expects it.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from llmapp.security.redact import Redactor

TRACER_NAME = "llmapp"

# Content is off by default. Turning it on is a privacy
# decision, not a debugging convenience.
RECORD_CONTENT = False


# OpenTelemetry's global provider can only be set once per
# process, which makes it unusable for tests that need a
# fresh exporter. The application keeps its own reference.
_provider: TracerProvider | None = None


def configure_tracing(
    *, install_globally: bool = False
) -> InMemorySpanExporter:
    """Set up tracing and return an exporter for tests."""
    global _provider
    _provider = TracerProvider()
    exporter = InMemorySpanExporter()
    _provider.add_span_processor(SimpleSpanProcessor(exporter))
    if install_globally:
        trace.set_tracer_provider(_provider)
    return exporter


def tracer() -> trace.Tracer:
    if _provider is not None:
        return _provider.get_tracer(TRACER_NAME)
    return trace.get_tracer(TRACER_NAME)


@dataclass(frozen=True)
class LlmCall:
    """The facts about one model call worth recording."""

    system: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    finish_reason: str = ""
    tier: str = "primary"
    cached: bool = False


@contextmanager
def request_span(
    *,
    feature: str,
    principal: str,
    correlation_id: str,
) -> Iterator[trace.Span]:
    """The root span. Identifiers only, never content."""
    with tracer().start_as_current_span("llm.request") as span:
        span.set_attribute("llmapp.feature", feature)
        span.set_attribute("llmapp.principal", principal)
        span.set_attribute("llmapp.correlation_id", correlation_id)
        yield span


@contextmanager
def retrieval_span(
    *, query_tokens: int, k: int
) -> Iterator[trace.Span]:
    with tracer().start_as_current_span("llm.retrieval") as span:
        span.set_attribute("llmapp.retrieval.k", k)
        span.set_attribute(
            "llmapp.retrieval.query_tokens", query_tokens
        )
        yield span


def record_hits(
    span: trace.Span,
    chunk_ids: Sequence[str],
    scores: Sequence[float],
) -> None:
    """Chunk ids are the incident-response primitive."""
    span.set_attribute(
        "llmapp.retrieval.chunk_ids", list(chunk_ids)
    )
    span.set_attribute("llmapp.retrieval.count", len(chunk_ids))
    if scores:
        span.set_attribute(
            "llmapp.retrieval.top_score", float(scores[0])
        )


@contextmanager
def model_span(operation: str = "chat") -> Iterator[trace.Span]:
    with tracer().start_as_current_span(
        f"llm.{operation}"
    ) as span:
        yield span


def record_call(span: trace.Span, call: LlmCall) -> None:
    span.set_attribute("gen_ai.system", call.system)
    span.set_attribute("gen_ai.request.model", call.model)
    span.set_attribute(
        "gen_ai.usage.input_tokens", call.input_tokens
    )
    span.set_attribute(
        "gen_ai.usage.output_tokens", call.output_tokens
    )
    span.set_attribute("llmapp.cost_usd", round(call.cost_usd, 6))
    span.set_attribute("llmapp.tier", call.tier)
    span.set_attribute("llmapp.cached", call.cached)
    if call.finish_reason:
        span.set_attribute(
            "gen_ai.response.finish_reason", call.finish_reason
        )


@contextmanager
def tool_span(name: str) -> Iterator[trace.Span]:
    with tracer().start_as_current_span("llm.tool") as span:
        span.set_attribute("llmapp.tool.name", name)
        yield span


def record_content(
    span: trace.Span,
    key: str,
    text: str,
    *,
    enabled: bool = RECORD_CONTENT,
    redactor: Redactor | None = None,
) -> bool:
    """Attach content only when explicitly enabled.

    Returns whether anything was written, so a caller can
    assert on the default rather than assume it.
    """
    if not enabled:
        return False
    scrubber = redactor or Redactor()
    span.set_attribute(key, scrubber.scrub(text))
    return True


def waterfall(exporter: InMemorySpanExporter) -> str:
    """Render finished spans as a readable trace."""
    spans = list(exporter.get_finished_spans())
    by_id: dict[int, Any] = {
        span.context.span_id: span for span in spans
    }
    depth: dict[int, int] = {}
    for span in spans:
        level = 0
        parent = span.parent
        while parent is not None and parent.span_id in by_id:
            level += 1
            parent = by_id[parent.span_id].parent
        depth[span.context.span_id] = level
    ordered = sorted(spans, key=lambda s: s.start_time or 0)
    lines = []
    for span in ordered:
        micros = (
            (span.end_time or 0) - (span.start_time or 0)
        ) / 1000
        pad = "  " * depth[span.context.span_id]
        lines.append(f"{pad}{span.name:<24} {micros:8.1f} us")
    return "\n".join(lines)
