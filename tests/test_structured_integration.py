import json

import pytest

from llmapp.llm.base import Message, SupportsStructuredOutput
from llmapp.llm.openai_adapter import OpenAIClient
from llmapp.schemas.strict import strict_schema
from llmapp.schemas.triage import TicketExtraction
from llmapp.schemas.validate import parse_or_repair
from tests._mock_schema import SEEN, STATE, serve

ASK = [Message(role="user", content="I was charged twice")]


@pytest.fixture(scope="module")
def client() -> OpenAIClient:
    serve(8094)
    return OpenAIClient(
        api_key="test",
        model="mock-model",
        base_url="http://127.0.0.1:8094/v1",
    )


@pytest.mark.local_server
def test_adapter_advertises_structured_output(
    client: OpenAIClient,
) -> None:
    assert isinstance(client, SupportsStructuredOutput)


@pytest.mark.local_server
def test_the_schema_reaches_the_provider(
    client: OpenAIClient,
) -> None:
    SEEN.clear()
    STATE["replies"] = [json.dumps(VALID)]
    client.complete_json(
        ASK,
        schema=strict_schema(TicketExtraction),
        schema_name="ticket_extraction",
        max_output_tokens=256,
    )
    sent = SEEN[0]["text"]["format"]
    assert sent["type"] == "json_schema"
    assert sent["name"] == "ticket_extraction"
    assert sent["strict"] is True
    assert sent["schema"]["additionalProperties"] is False


@pytest.mark.local_server
def test_repair_loop_over_the_real_client(
    client: OpenAIClient,
) -> None:
    STATE["replies"] = [
        '{"category": "finance"}',
        json.dumps(VALID),
    ]
    parsed, report = parse_or_repair(
        client, ASK, TicketExtraction, attempts=2
    )
    assert parsed.category.value == "billing"
    assert report.attempts == 2
    assert report.first_pass_valid is False


VALID = {
    "category": "billing",
    "severity": "high",
    "order": {"order_id": "48812", "amount_usd": 42.5},
    "summary": "Charged twice for one order.",
    "needs_human": True,
    "confidence": 0.9,
}
