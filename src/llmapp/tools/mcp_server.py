"""Expose the local tool registry over MCP.

Note: this is the mcp 2.x API. In 1.x the server class was
called FastMCP; importing that name here raises an error
pointing at the migration guide.
"""

import inspect
import json
from typing import Any

from mcp.server.mcpserver import MCPServer

from llmapp.tools.base import ToolCall, ToolSpec
from llmapp.tools.registry import ToolRegistry


def _handler(registry: ToolRegistry, spec: ToolSpec) -> Any:
    """Wrap one local tool as an MCP-callable function.

    The server derives the advertised schema by inspecting
    the function's signature, so the signature is built from
    the tool's Pydantic model rather than left as **kwargs.
    """

    def call(**kwargs: Any) -> str:
        result = registry.dispatch(
            ToolCall(
                call_id=f"mcp-{spec.name}",
                name=spec.name,
                arguments=json.dumps(kwargs),
            ),
            approved=[f"mcp-{spec.name}"],
        )
        return result.output

    parameters = []
    annotations: dict[str, Any] = {"return": str}
    for name, field in spec.arguments.model_fields.items():
        annotation = field.annotation or str
        default = (
            inspect.Parameter.empty
            if field.is_required()
            else field.get_default(call_default_factory=True)
        )
        parameters.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=annotation,
            )
        )
        annotations[name] = annotation

    call.__name__ = spec.name
    call.__doc__ = spec.description
    call.__annotations__ = annotations
    call.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters, return_annotation=str
    )
    return call


def build_server(
    registry: ToolRegistry,
    *,
    name: str = "llmapp-tools",
    expose: list[str] | None = None,
) -> MCPServer:
    """One server, advertising an explicit list of tools."""
    server = MCPServer(name=name)
    names = expose if expose is not None else sorted(registry.specs)
    for tool_name in names:
        spec = registry.specs[tool_name]
        server.add_tool(
            _handler(registry, spec),
            name=spec.name,
            description=spec.description,
        )
    return server
