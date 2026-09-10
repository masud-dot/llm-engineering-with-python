import pytest

from llmapp.llm.anthropic_adapter import AnthropicClient
from llmapp.llm.base import Message
from tests._mock_anthropic import SEEN, serve

CONVERSATION = [
    Message(role="system", content="Classify the ticket."),
    Message(role="user", content="I was charged twice"),
]


@pytest.fixture(scope="module")
def client() -> AnthropicClient:
    serve(8098)
    return AnthropicClient(
        api_key="test",
        model="mock-model",
        base_url="http://127.0.0.1:8098",
    )


@pytest.mark.local_server
def test_same_completion_shape_as_openai(
    client: AnthropicClient,
) -> None:
    SEEN.clear()
    result = client.complete(CONVERSATION, max_output_tokens=64)
    assert result.text == "billing"
    assert result.input_tokens == 37
    assert result.finish_reason == "end_turn"


@pytest.mark.local_server
def test_system_is_a_top_level_field(
    client: AnthropicClient,
) -> None:
    SEEN.clear()
    client.complete(CONVERSATION, max_output_tokens=64)
    sent = SEEN[0]
    assert sent["system"] == "Classify the ticket."
    assert sent["max_tokens"] == 64
    assert len(sent["messages"]) == 1
