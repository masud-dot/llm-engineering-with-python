"""Consume tools from an MCP server as local ToolSpecs."""

import json
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator, Sequence
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from pydantic import BaseModel, create_model

TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def model_from_schema(
    name: str, schema: dict[str, Any]
) -> type[BaseModel]:
    """Build a Pydantic model from a remote tool's schema.

    Only flat, scalar properties are mapped; anything else
    falls back to a permissive type. The remote server
    validates too, so this is a first line rather than the
    only one.
    """
    fields: dict[str, Any] = {}
    required = set(schema.get("required", []))
    for key, prop in schema.get("properties", {}).items():
        python_type = TYPES.get(prop.get("type", ""), Any)
        default = ... if key in required else None
        annotation = (
            python_type if key in required else python_type | None
        )
        fields[key] = (annotation, default)
    return create_model(name, **fields)


@asynccontextmanager
async def connect(
    command: str,
    args: Sequence[str],
    env: dict[str, str] | None = None,
) -> AsyncIterator[ClientSession]:
    """Start a server as a child process and speak to it.

    The child does not inherit this process's environment
    unless you pass it. Omitting `env` is the most common
    reason a stdio server fails to start.
    """
    params = StdioServerParameters(
        command=command, args=list(args), env=env
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def discover(
    session: ClientSession,
) -> list[tuple[str, str, type[BaseModel]]]:
    """What the server offers, as local argument models."""
    listing = await session.list_tools()
    found = []
    for tool in listing.tools:
        # 2.x renamed inputSchema to input_schema.
        model = model_from_schema(
            f"{tool.name}_args", tool.input_schema or {}
        )
        found.append((tool.name, tool.description or "", model))
    return found


async def call_remote(
    session: ClientSession, name: str, arguments: dict[str, Any]
) -> str:
    """Call a remote tool and flatten the result to text."""
    result = await session.call_tool(name, arguments)
    parts = []
    for item in getattr(result, "content", []):
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
    if not parts:
        return json.dumps(
            getattr(result, "structuredContent", {}) or {}
        )
    return "\n".join(parts)
