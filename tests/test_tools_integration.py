import pytest

from llmapp.llm.base import Message, SupportsTools
from llmapp.llm.openai_adapter import OpenAIClient
from llmapp.tools.loop import results_as_input, run_calls
from tests._mock_tools import SEEN, STATE, serve
from tests.test_tools import registry

ASK = [Message(role="user", content="what is the refund window")]


@pytest.fixture(scope="module")
def client() -> OpenAIClient:
    serve(8092)
    return OpenAIClient(
        api_key="test",
        model="mock-model",
        base_url="http://127.0.0.1:8092/v1",
    )


@pytest.mark.local_server
def test_the_adapter_advertises_tool_support(
    client: OpenAIClient,
) -> None:
    assert isinstance(client, SupportsTools)


@pytest.mark.local_server
def test_tool_definitions_reach_the_wire_flat(
    client: OpenAIClient,
) -> None:
    SEEN.clear()
    STATE["mode"] = "call"
    client.complete_with_tools(
        ASK, tools=registry().schemas(["search_docs"])
    )
    sent = SEEN[0]["tools"][0]
    assert sent["type"] == "function"
    assert sent["name"] == "search_docs"
    assert "function" not in sent
    assert sent["parameters"]["additionalProperties"] is False


@pytest.mark.local_server
def test_a_function_call_is_parsed_from_the_response(
    client: OpenAIClient,
) -> None:
    STATE["mode"] = "call"
    completion = client.complete_with_tools(
        ASK, tools=registry().schemas()
    )
    assert len(completion.tool_calls) == 1
    request = completion.tool_calls[0]
    assert request.name == "search_docs"
    assert request.call_id == "call_abc"
    assert "refund window" in request.arguments


@pytest.mark.local_server
def test_the_full_turn_sends_results_back(
    client: OpenAIClient,
) -> None:
    STATE["mode"] = "call"
    first = client.complete_with_tools(
        ASK, tools=registry().schemas()
    )
    results = run_calls(registry(), first.tool_calls)
    assert results[0].ok
    assert "3 results for refund window" in results[0].output

    SEEN.clear()
    STATE["mode"] = "text"
    second = client.complete_with_tools(
        ASK,
        tools=registry().schemas(),
        extra_input=results_as_input(first.tool_calls, results),
    )
    payload = SEEN[0]["input"]
    kinds = [item.get("type") for item in payload]
    assert "function_call" in kinds
    assert "function_call_output" in kinds
    assert second.text == "Refunds take fourteen days."
    assert second.tool_calls == ()
