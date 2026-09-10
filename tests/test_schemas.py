import json

import pytest
from pydantic import ValidationError

from llmapp.llm.base import Completion, Message
from llmapp.llm.errors import SemanticError
from llmapp.schemas.strict import strict_schema
from llmapp.schemas.triage import Category, TicketExtraction
from llmapp.schemas.validate import (
    extract_json,
    parse_or_repair,
)

ASK = [Message(role="user", content="I was charged twice")]

GOOD = json.dumps(
    {
        "category": "billing",
        "severity": "high",
        "order": {"order_id": "48812", "amount_usd": 42.5},
        "summary": "Charged twice for one order.",
        "needs_human": True,
        "confidence": 0.9,
    }
)

BAD_ENUM = GOOD.replace('"billing"', '"finance"')
BAD_RANGE = GOOD.replace("0.9", "1.7")
NOT_JSON = "Sure! The category is billing."
FENCED = f"```json\n{GOOD}\n```"


class Replies:
    """Return canned strings, recording what it was sent."""

    def __init__(self, *texts: str, finish: str = "stop") -> None:
        self.texts = list(texts)
        self.finish = finish
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: object,
        *,
        max_output_tokens: int = 512,
    ) -> Completion:
        self.calls.append(list(messages))  # type: ignore[arg-type]
        return Completion(
            self.texts[len(self.calls) - 1], 10, 5, "x", self.finish
        )


def test_strict_schema_forbids_extras_everywhere() -> None:
    schema = strict_schema(TicketExtraction)
    assert schema["additionalProperties"] is False
    order = schema["$defs"]["OrderRef"]
    assert order["additionalProperties"] is False
    assert set(order["required"]) == {"order_id", "amount_usd"}


def test_optional_fields_are_still_required_by_name() -> None:
    schema = strict_schema(TicketExtraction)
    assert "order" in schema["required"]


def test_valid_payload_parses() -> None:
    client = Replies(GOOD)
    parsed, report = parse_or_repair(
        client, ASK, TicketExtraction
    )
    assert parsed.category is Category.BILLING
    assert parsed.order is not None
    assert parsed.order.order_id == "48812"
    assert report.first_pass_valid
    assert report.attempts == 1


def test_fenced_json_is_recovered() -> None:
    parsed, _ = parse_or_repair(
        Replies(FENCED), ASK, TicketExtraction
    )
    assert parsed.severity.value == "high"


def test_invalid_enum_is_repaired_on_the_second_try() -> None:
    client = Replies(BAD_ENUM, GOOD)
    parsed, report = parse_or_repair(
        client, ASK, TicketExtraction
    )
    assert parsed.category is Category.BILLING
    assert report.attempts == 2
    assert report.first_pass_valid is False


def test_the_error_is_shown_to_the_model() -> None:
    client = Replies(BAD_RANGE, GOOD)
    parse_or_repair(client, ASK, TicketExtraction)
    repair_turn = client.calls[1][-1]
    assert repair_turn.role == "user"
    assert "confidence" in repair_turn.content
    assert "did not match the required schema" in (
        repair_turn.content
    )


def test_repair_attempts_are_bounded() -> None:
    client = Replies(BAD_ENUM, BAD_ENUM, BAD_ENUM)
    with pytest.raises(SemanticError, match="after 2 attempts"):
        parse_or_repair(client, ASK, TicketExtraction, attempts=2)
    assert len(client.calls) == 2


def test_prose_instead_of_json_fails_loudly() -> None:
    client = Replies(NOT_JSON)
    with pytest.raises(SemanticError):
        parse_or_repair(client, ASK, TicketExtraction, attempts=1)


def test_truncation_is_not_a_validation_problem() -> None:
    client = Replies(GOOD[:40], finish="incomplete")
    with pytest.raises(SemanticError, match="truncated"):
        parse_or_repair(client, ASK, TicketExtraction)
    assert len(client.calls) == 1


def test_schema_rejects_extra_fields_locally() -> None:
    payload = json.loads(GOOD) | {"invented": "yes"}
    with pytest.raises(ValidationError):
        TicketExtraction.model_validate(payload)


def test_summary_is_normalized_by_the_validator() -> None:
    payload = json.loads(GOOD)
    payload["summary"] = "  Charged\n  twice.  "
    parsed = TicketExtraction.model_validate(payload)
    assert parsed.summary == "Charged twice."


def test_extract_json_handles_surrounding_prose() -> None:
    assert extract_json('Here you go: {"a": 1} hope that helps')


def test_semantic_errors_are_not_retried_by_the_reliability_layer(
) -> None:
    """A schema failure is not a transport failure."""
    from llmapp.llm.pricing import Rates
    from llmapp.llm.reliable import ReliableClient

    class AlwaysSemantic:
        def __init__(self) -> None:
            self.calls = 0

        def complete(
            self,
            messages: object,
            *,
            max_output_tokens: int = 512,
        ) -> Completion:
            self.calls += 1
            raise SemanticError("schema violation")

    inner = AlwaysSemantic()
    client = ReliableClient(
        primary=inner,  # type: ignore[arg-type]
        rates=Rates(3.0, 15.0),
        sleep=lambda _s: None,
    )
    with pytest.raises(SemanticError):
        client.complete(ASK)
    assert inner.calls == 1
