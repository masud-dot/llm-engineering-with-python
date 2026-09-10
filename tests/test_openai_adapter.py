import pytest

from llmapp.llm.base import Message
from llmapp.llm.openai_adapter import OpenAIClient, split_system
from tests._mock_openai import SEEN, serve

CONVERSATION = [
    Message(role="system", content="Classify the ticket."),
    Message(role="user", content="I was charged twice"),
]


def test_system_turns_move_to_instructions() -> None:
    system, turns = split_system(CONVERSATION)
    assert system == "Classify the ticket."
    assert len(turns) == 1
    assert turns[0]["role"] == "user"


@pytest.fixture(scope="module")
def client() -> OpenAIClient:
    serve(8099)
    return OpenAIClient(
        api_key="test",
        model="mock-model",
        base_url="http://127.0.0.1:8099/v1",
    )


@pytest.mark.local_server
def test_complete_reads_text_and_usage(
    client: OpenAIClient,
) -> None:
    SEEN.clear()
    result = client.complete(CONVERSATION, max_output_tokens=64)
    assert result.text == "billing"
    assert result.input_tokens == 42
    assert result.model == "mock-model"


@pytest.mark.local_server
def test_request_carries_the_right_fields(
    client: OpenAIClient,
) -> None:
    SEEN.clear()
    client.complete(CONVERSATION, max_output_tokens=64)
    sent = SEEN[0]
    assert sent["model"] == "mock-model"
    assert sent["instructions"] == "Classify the ticket."
    assert sent["max_output_tokens"] == 64
    assert sent["input"][0]["content"] == "I was charged twice"


@pytest.mark.local_server
def test_stream_yields_deltas(client: OpenAIClient) -> None:
    pieces = list(client.stream(CONVERSATION))
    assert pieces == ["bil", "ling"]
