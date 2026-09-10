import asyncio

from llmapp.llm.base import (
    Message,
    SupportsAsync,
    SupportsStreaming,
)
from llmapp.llm.fake import ScriptedClient

ASK = [Message(role="user", content="hello")]


def test_scripted_client_advertises_capabilities() -> None:
    client = ScriptedClient(["a b c"])
    assert isinstance(client, SupportsStreaming)
    assert isinstance(client, SupportsAsync)


def test_stream_yields_pieces_that_join_to_the_text() -> None:
    client = ScriptedClient(["one two three"])
    pieces = list(client.stream(ASK))
    assert len(pieces) == 3
    assert "".join(pieces).strip() == "one two three"


def test_concurrent_calls_all_return() -> None:
    client = ScriptedClient(["a", "b", "c"])

    async def run() -> list[str]:
        results = await asyncio.gather(
            *(client.acomplete(ASK) for _ in range(3))
        )
        return [r.text for r in results]

    assert sorted(asyncio.run(run())) == ["a", "b", "c"]
