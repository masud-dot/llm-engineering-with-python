import asyncio
import os
import sys

import pytest

from llmapp.tools.mcp_client import (
    call_remote,
    connect,
    discover,
    model_from_schema,
)

SERVER = ["-m", "scripts.mcp_tools_server"]
# The child process gets nothing it is not given.
ENV = {
    "PATH": os.environ.get("PATH", ""),
    "PYTHONPATH": f"src{os.pathsep}.",
}


def test_a_flat_schema_becomes_a_model() -> None:
    model = model_from_schema(
        "Args",
        {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["topic"],
        },
    )
    parsed = model.model_validate({"topic": "refunds"})
    assert parsed.topic == "refunds"  # type: ignore[attr-defined]
    with pytest.raises(Exception):
        model.model_validate({"limit": 3})


@pytest.mark.local_server
def test_tools_are_discovered_over_the_protocol() -> None:
    async def run() -> list[str]:
        async with connect(
                sys.executable, SERVER, ENV
            ) as session:
            found = await discover(session)
            return [name for name, _, _ in found]

    assert "lookup_policy" in asyncio.run(run())


@pytest.mark.local_server
def test_the_description_survives_the_round_trip() -> None:
    async def run() -> str:
        async with connect(
                sys.executable, SERVER, ENV
            ) as session:
            found = await discover(session)
            return next(
                desc
                for name, desc, _ in found
                if name == "lookup_policy"
            )

    assert "policy statement" in asyncio.run(run())


@pytest.mark.local_server
def test_a_remote_tool_runs_and_returns_text() -> None:
    async def run() -> str:
        async with connect(
                sys.executable, SERVER, ENV
            ) as session:
            return await call_remote(
                session,
                "lookup_policy",
                {"topic": "refund window"},
            )

    assert "fourteen days" in asyncio.run(run())


@pytest.mark.local_server
def test_the_remote_schema_matches_the_local_one() -> None:
    async def run() -> dict[str, object]:
        async with connect(
                sys.executable, SERVER, ENV
            ) as session:
            found = await discover(session)
            _, _, model = next(
                item for item in found if item[0] == "lookup_policy"
            )
            return dict(model.model_fields)

    assert "topic" in asyncio.run(run())
