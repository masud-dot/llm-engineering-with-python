import pytest

from llmapp.llm.base import Message
from llmapp.llm.errors import PermanentError, TransientError
from llmapp.llm.openai_adapter import OpenAIClient
from llmapp.llm.pricing import Rates
from llmapp.llm.reliable import ReliableClient, RetryPolicy
from tests._mock_flaky import STATE, serve

ASK = [Message(role="user", content="hello")]
RATES = Rates(input_per_mtok=3.0, output_per_mtok=15.0)


@pytest.fixture(scope="module")
def adapter() -> OpenAIClient:
    serve(8096)
    return OpenAIClient(
        api_key="test",
        model="mock-model",
        base_url="http://127.0.0.1:8096/v1",
    )


@pytest.mark.local_server
def test_503_becomes_a_transient_error(
    adapter: OpenAIClient,
) -> None:
    STATE.update(fail_next=1, status=503, calls=0)
    with pytest.raises(TransientError):
        adapter.complete(ASK)
    assert STATE["calls"] == 1  # sdk_retries=0, so exactly one


@pytest.mark.local_server
def test_400_becomes_a_permanent_error(
    adapter: OpenAIClient,
) -> None:
    STATE.update(fail_next=1, status=400, calls=0)
    with pytest.raises(PermanentError):
        adapter.complete(ASK)


@pytest.mark.local_server
def test_wrapper_survives_injected_failures(
    adapter: OpenAIClient,
) -> None:
    STATE.update(fail_next=2, status=503, calls=0)
    client = ReliableClient(
        primary=adapter,
        rates=RATES,
        policy=RetryPolicy(attempts=4),
        sleep=lambda _s: None,
    )
    result = client.complete(ASK)
    assert result.text == "ok"
    assert STATE["calls"] == 3


@pytest.mark.local_server
def test_retry_after_header_is_honored(
    adapter: OpenAIClient,
) -> None:
    STATE.update(
        fail_next=1, status=429, retry_after=7, calls=0
    )
    waits: list[float] = []
    client = ReliableClient(
        primary=adapter,
        rates=RATES,
        sleep=waits.append,
    )
    client.complete(ASK)
    assert waits == [7.0]
    STATE["retry_after"] = None
