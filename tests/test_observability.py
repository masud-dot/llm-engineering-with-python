import pytest

from llmapp.eval.dataset import Case
from llmapp.obs.feedback import (
    Feedback,
    FeedbackStore,
    Verdict,
)
from llmapp.obs.metrics import (
    Metrics,
    Rule,
    Series,
    Severity,
    change_rate,
    evaluate,
)
from llmapp.obs.tracing import (
    LlmCall,
    configure_tracing,
    model_span,
    record_call,
    record_content,
    record_hits,
    request_span,
    retrieval_span,
    tool_span,
    waterfall,
)


@pytest.fixture
def exporter():
    return configure_tracing()


def one_request() -> None:
    with request_span(
        feature="rag", principal="u1", correlation_id="c1"
    ):
        with retrieval_span(query_tokens=9, k=3) as span:
            record_hits(
                span,
                ["refunds.md#0", "escalation.pdf#0"],
                [0.61, 0.20],
            )
        with tool_span("search_corpus"):
            pass
        with model_span("chat") as span:
            record_call(
                span,
                LlmCall(
                    system="openai",
                    model="mock-model",
                    input_tokens=4400,
                    output_tokens=120,
                    cost_usd=0.0150,
                    finish_reason="stop",
                ),
            )


# --- tracing --------------------------------------------------


def test_a_request_produces_a_nested_trace(exporter) -> None:
    one_request()
    names = [s.name for s in exporter.get_finished_spans()]
    assert set(names) == {
        "llm.request",
        "llm.retrieval",
        "llm.tool",
        "llm.chat",
    }


def test_child_spans_have_the_request_as_an_ancestor(
    exporter,
) -> None:
    one_request()
    spans = {s.name: s for s in exporter.get_finished_spans()}
    root = spans["llm.request"]
    assert root.parent is None
    for name in ("llm.retrieval", "llm.tool", "llm.chat"):
        assert spans[name].parent is not None


def test_chunk_ids_are_recorded_for_incident_response(
    exporter,
) -> None:
    one_request()
    span = next(
        s
        for s in exporter.get_finished_spans()
        if s.name == "llm.retrieval"
    )
    assert list(span.attributes["llmapp.retrieval.chunk_ids"]) == [
        "refunds.md#0",
        "escalation.pdf#0",
    ]
    assert span.attributes["llmapp.retrieval.top_score"] == (
        pytest.approx(0.61)
    )


def test_model_attributes_use_the_gen_ai_namespace(
    exporter,
) -> None:
    one_request()
    span = next(
        s
        for s in exporter.get_finished_spans()
        if s.name == "llm.chat"
    )
    assert span.attributes["gen_ai.system"] == "openai"
    assert span.attributes["gen_ai.usage.input_tokens"] == 4400
    assert span.attributes["llmapp.cost_usd"] == pytest.approx(
        0.015
    )


def test_the_principal_is_on_the_root_span(exporter) -> None:
    one_request()
    root = next(
        s
        for s in exporter.get_finished_spans()
        if s.name == "llm.request"
    )
    assert root.attributes["llmapp.principal"] == "u1"
    assert root.attributes["llmapp.correlation_id"] == "c1"


def test_content_is_not_recorded_by_default(exporter) -> None:
    with model_span() as span:
        wrote = record_content(
            span, "llmapp.prompt", "a@b.com asked something"
        )
    assert wrote is False
    finished = exporter.get_finished_spans()[0]
    assert "llmapp.prompt" not in finished.attributes


def test_content_is_redacted_when_enabled(exporter) -> None:
    with model_span() as span:
        wrote = record_content(
            span,
            "llmapp.prompt",
            "contact a@b.com about it",
            enabled=True,
        )
    assert wrote is True
    finished = exporter.get_finished_spans()[0]
    stored = finished.attributes["llmapp.prompt"]
    assert "a@b.com" not in stored
    assert "[redacted:email]" in stored


def test_the_waterfall_shows_nesting(exporter) -> None:
    one_request()
    rendered = waterfall(exporter)
    assert "llm.request" in rendered
    assert "  llm.retrieval" in rendered


# --- metrics --------------------------------------------------


def test_percentiles_come_from_the_distribution() -> None:
    series = Series("latency_ms")
    for value in range(1, 101):
        series.observe(float(value))
    assert series.percentile(0.5) == pytest.approx(50, abs=1)
    assert series.percentile(0.95) == pytest.approx(95, abs=1)
    assert series.mean == pytest.approx(50.5)


def test_an_empty_series_does_not_raise() -> None:
    assert Series("x").percentile(0.99) == 0.0


def test_rates_are_computed_from_counters() -> None:
    metrics = Metrics()
    for _ in range(10):
        metrics.count("requests")
    metrics.count("abstained", 3)
    metrics.count("errors")
    assert metrics.rate("abstained", "requests") == 0.3
    assert metrics.rate("errors", "requests") == 0.1
    assert metrics.rate("missing", "requests") == 0.0


def test_a_summary_line_is_produced() -> None:
    metrics = Metrics()
    metrics.count("requests", 4)
    metrics.latency_ms.observe(120)
    metrics.cost_usd.observe(0.02)
    assert "p50" in metrics.summary()


# --- alerting -------------------------------------------------


def test_a_ceiling_rule_fires() -> None:
    rule = Rule("cost_usd", value=0.05, ceiling=0.03)
    assert rule.fires()
    assert "above 0.03" in rule.describe()


def test_a_floor_rule_fires() -> None:
    rule = Rule("grounded_rate", value=0.70, floor=0.90)
    assert rule.fires()
    assert "below 0.9" in rule.describe()


def test_a_rule_inside_its_bounds_is_quiet() -> None:
    assert not Rule("cost_usd", value=0.01, ceiling=0.03).fires()


def test_pages_are_ordered_before_warnings() -> None:
    firing = evaluate(
        [
            Rule("cost", 1.0, ceiling=0.5),
            Rule(
                "errors",
                0.5,
                ceiling=0.1,
                severity=Severity.PAGE,
            ),
            Rule("latency", 10.0, ceiling=100.0),
        ]
    )
    assert [r.name for r in firing] == ["errors", "cost"]


def test_the_derivative_is_measurable() -> None:
    assert change_rate(0.10, 0.14) == pytest.approx(0.4)
    assert change_rate(0.0, 0.0) == 0.0
    assert change_rate(0.0, 0.5) == 1.0


# --- feedback -------------------------------------------------


def store() -> FeedbackStore:
    store = FeedbackStore()
    store.add(
        Feedback("r1", "refund window", Verdict.GOOD)
    )
    store.add(
        Feedback(
            "r2",
            "platinum refunds",
            Verdict.WRONG_FACT,
            retrieved_ids=("refunds.md#0",),
        )
    )
    store.add(
        Feedback("r3", "vendor terms", Verdict.MISSING)
    )
    return store


def test_satisfaction_is_a_rate() -> None:
    assert store().satisfaction() == pytest.approx(1 / 3)


def test_negative_feedback_becomes_evaluation_cases() -> None:
    cases = store().to_cases()
    assert [c.id for c in cases] == ["prod-r2", "prod-r3"]
    assert all(c.source == "production" for c in cases)
    assert isinstance(cases[0], Case)


def test_a_missing_answer_case_is_marked_unanswerable(
) -> None:
    cases = {c.id: c for c in store().to_cases()}
    assert cases["prod-r3"].answerable is False
    assert cases["prod-r2"].answerable is True


def test_retrieved_ids_are_carried_into_the_case() -> None:
    cases = {c.id: c for c in store().to_cases()}
    assert cases["prod-r2"].relevant_ids == ("refunds.md#0",)
